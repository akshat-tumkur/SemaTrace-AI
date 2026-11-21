"""Pinned MiniLM ONNX encoder. Lazy, CPU-only and strictly local-file loading."""

from __future__ import annotations

from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
from threading import RLock

LOCK_PATH = Path(__file__).resolve().parents[2] / "corpus/minilm-onnx.lock.json"
PREPROCESSING = "minilm-onnx-mean-l2-256-v1"
_load_lock = RLock()


class LocalModelUnavailable(RuntimeError):
    pass


def model_lock():
    return json.loads(LOCK_PATH.read_text(encoding="utf-8"))


def default_directory():
    return Path(
        os.getenv("SEMATRACE_LOCAL_MODEL_DIR")
        or Path(__file__).resolve().parents[2]
        / "data/models/all-MiniLM-L6-v2"
        / model_lock()["revision"]
    )


def verify_artifact(raw, entry):
    if len(raw) != entry["size"]:
        raise LocalModelUnavailable("local_model_size_mismatch")
    if "lfs" in entry:
        valid = hashlib.sha256(raw).hexdigest() == entry["lfs"]["sha256"]
    else:
        valid = (
            hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
            == entry["blobId"]
        )
    if not valid:
        raise LocalModelUnavailable("local_model_hash_mismatch")


def verify_directory(directory):
    lock = model_lock()
    try:
        for entry in lock["files"]:
            verify_artifact((directory / entry["rfilename"]).read_bytes(), entry)
        config = json.loads((directory / "config.json").read_text())
        pooling = json.loads((directory / "1_Pooling/config.json").read_text())
        sentence = json.loads((directory / "sentence_bert_config.json").read_text())
        if (
            config["hidden_size"] != 384
            or sentence["max_seq_length"] != 256
            or not pooling["pooling_mode_mean_tokens"]
            or pooling["pooling_mode_cls_token"]
            or pooling["pooling_mode_max_tokens"]
        ):
            raise LocalModelUnavailable("local_model_architecture_mismatch")
    except (OSError, ValueError, KeyError) as e:
        raise LocalModelUnavailable("local_model_files_missing_or_invalid") from e
    return lock


class MiniLMEncoder:
    def __init__(self, directory):
        self.identity = verify_directory(Path(directory))
        try:
            import numpy as np
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError as e:
            raise LocalModelUnavailable("install_requirements_local_txt") from e
        self.np = np
        self.tokenizer = Tokenizer.from_file(str(Path(directory) / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=256)
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        options.enable_cpu_mem_arena = False
        self.session = ort.InferenceSession(
            str(Path(directory) / "onnx/model.onnx"),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.session.disable_fallback()
        self.lock = RLock()
        self.calls = 0
        self.truncated_passages = 0
        self.runtime_version = ort.__version__

    def encode(self, texts):
        try:
            return self._encode(texts)
        except LocalModelUnavailable:
            raise
        except Exception as e:
            raise LocalModelUnavailable(
                "local_inference_failed:" + type(e).__name__
            ) from e

    def _encode(self, texts):
        if not texts:
            return []
        if len(texts) > 64 or any(
            not isinstance(t, str) or len(t) > 100_000 for t in texts
        ):
            raise LocalModelUnavailable("local_batch_or_passage_limit")
        with self.lock:
            encoded = self.tokenizer.encode_batch(texts)
            self.truncated_passages += sum(bool(e.overflowing) for e in encoded)
            np = self.np
            feed = {
                "input_ids": np.asarray([e.ids for e in encoded], dtype=np.int64),
                "attention_mask": np.asarray(
                    [e.attention_mask for e in encoded], dtype=np.int64
                ),
                "token_type_ids": np.asarray(
                    [e.type_ids for e in encoded], dtype=np.int64
                ),
            }
            names = {i.name for i in self.session.get_inputs()}
            output = self.session.run(
                None, {k: v for k, v in feed.items() if k in names}
            )[0]
            if (
                output.shape != (len(texts), len(encoded[0].ids), 384)
                or not np.isfinite(output).all()
            ):
                raise LocalModelUnavailable("local_token_vector_shape")
            mask = feed["attention_mask"][..., None].astype(np.float32)
            pooled = (output * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1e-9)
            norms = np.linalg.norm(pooled, axis=1, keepdims=True)
            if not np.isfinite(norms).all() or (norms <= 0).any():
                raise LocalModelUnavailable("local_nonfinite_or_zero_vector")
            self.calls += 1
            return (pooled / norms).astype(np.float32).tolist()


@lru_cache(maxsize=1)
def _load(directory):
    return MiniLMEncoder(directory)


def local_encoder():
    # Serialize cold initialization as lru_cache alone permits duplicate misses.
    with _load_lock:
        try:
            return _load(str(default_directory().resolve()))
        except LocalModelUnavailable:
            raise
        except Exception as e:
            raise LocalModelUnavailable(
                "local_model_initialization_failed:" + type(e).__name__
            ) from e
