"""Optional private CPU RoBERTa. No Torch, downloads, or classification threshold."""

import hashlib, json, os
from pathlib import Path
from threading import RLock
from functools import lru_cache
import numpy as np

CHECKPOINT = "c5c38ed47631933df1e2fa6069e2247cc0310630ad1b14821bdc0b8b581d636b"
MODEL_SHA = "7c48d07cf73562b04af6af55cb4b27d62c7f6685fc9463cb83ae48f784ec7c9f"
TOKENIZER_SHA = "12e18fdf23435245cf160bfa464e3d140977dec264fca73adf5c05307d28cd28"
MAX_PAIRS = 8
_lock = RLock()


class RerankerUnavailable(RuntimeError):
    pass


def directory():
    value = os.getenv("SEMATRACE_REVIEW_MODEL_DIR", "")
    if not value:
        raise RerankerUnavailable("Optional private review model is not configured.")
    return str(Path(value).resolve())


class ReviewReranker:
    def __init__(self, path):
        from tokenizers import Tokenizer
        import onnxruntime as ort

        root = Path(path)
        try:
            self.manifest = json.loads((root / "review-manifest.json").read_text())
            m = self.manifest
            if (
                m["role"] != "review-reranker-v1"
                or m["checkpoint_sha256"] != CHECKPOINT
                or m["autonomous_classifier_accepted"] is not False
                or m["coverage_eligible"] is not False
                or m["score_is_probability"] is not False
            ):
                raise ValueError("Invalid review role")
            if not {"model.onnx", "tokenizer.json"} <= m["files"].keys():
                raise ValueError("Missing model identities")
            if (
                m["files"]["model.onnx"] != MODEL_SHA
                or m["files"]["tokenizer.json"] != TOKENIZER_SHA
            ):
                raise ValueError("Unexpected exported review artifact")
            for name, expected in m["files"].items():
                if Path(name).name != name:
                    raise ValueError("Invalid artifact path")
                with (root / name).open("rb") as stream:
                    if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                        raise ValueError("Artifact hash mismatch")
            self.tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
            self.tokenizer.enable_truncation(max_length=512, strategy="longest_first")
            self.tokenizer.enable_padding(pad_id=1, pad_token="<pad>")
            options = ort.SessionOptions()
            options.intra_op_num_threads = 2
            options.inter_op_num_threads = 1
            options.enable_cpu_mem_arena = False
            self.session = ort.InferenceSession(
                str(root / "model.onnx"),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
            self.session.disable_fallback()
        except Exception as e:
            raise RerankerUnavailable(
                "Optional review model failed integrity or runtime validation."
            ) from e
        self.lock = RLock()
        self.cache = {}

    def score(self, pairs):
        if (
            not pairs
            or len(pairs) > MAX_PAIRS
            or any(len(s) > 10000 or len(t) > 10000 for s, t in pairs)
        ):
            raise ValueError(
                "Review is bounded to eight source/sentence pairs of at most 10000 characters."
            )
        with self.lock:
            keys = [
                hashlib.sha256((s + "\0" + t).encode()).hexdigest() for s, t in pairs
            ]
            missing = list(dict.fromkeys(k for k in keys if k not in self.cache))
            if missing:
                examples = [pairs[keys.index(k)] for k in missing]
                encoded = self.tokenizer.encode_batch(examples)
                try:
                    logits = self.session.run(
                        None,
                        {
                            "input_ids": np.asarray(
                                [e.ids for e in encoded], dtype=np.int64
                            ),
                            "attention_mask": np.asarray(
                                [e.attention_mask for e in encoded], dtype=np.int64
                            ),
                        },
                    )[0]
                except Exception as e:
                    raise RerankerUnavailable(
                        "CPU review inference failed; core evidence is unchanged."
                    ) from e
                if logits.shape != (len(missing), 2) or not np.isfinite(logits).all():
                    raise RerankerUnavailable("Invalid review model output")
                # Logit margin is monotonic with positive-head softmax, but is
                # deliberately not exposed as a calibrated reuse probability.
                for key, row, enc in zip(missing, logits, encoded):
                    self.cache[key] = {
                        "score": float(row[1] - row[0]),
                        "truncated": bool(enc.overflowing),
                        "tokens": len(enc.ids),
                    }
                while len(self.cache) > 256:
                    self.cache.pop(next(iter(self.cache)))
            return [dict(self.cache[k]) for k in keys]


@lru_cache(maxsize=1)
def _load(path):
    return ReviewReranker(path)


def review_model():
    with _lock:
        return _load(directory())


def release_review_model():
    """The serialized investigation frees optional ONNX memory before local LLMs."""
    with _lock:
        _load.cache_clear()
