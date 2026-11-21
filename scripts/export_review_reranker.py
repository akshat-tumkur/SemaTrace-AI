"""Private review-role export. Does NOT enable the failed autonomous classifier.

Run with the existing practical-training environment; no training or evaluation
partitions are opened. Output is deliberately outside the public repository.
"""

import argparse, hashlib, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

WEIGHTS = "c5c38ed47631933df1e2fa6069e2247cc0310630ad1b14821bdc0b8b581d636b"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--campaign", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    import torch, numpy as np, onnxruntime as ort
    from training.engine import load_best

    torch.set_num_threads(2)
    if sha(args.campaign / "best/model.pt") != WEIGHTS:
        raise ValueError("Unexpected trained checkpoint")
    if args.output.exists():
        raise ValueError(
            "Use a new private output directory; no artifact is overwritten"
        )
    model, tok, _ = load_best(args.campaign, "cpu")
    model.eval()
    # Explicit eager attention supports the legacy ONNX exporter in Torch2.6.
    model.roberta.set_attn_implementation("eager") if hasattr(
        model.roberta, "set_attn_implementation"
    ) else None
    args.output.mkdir(parents=True)
    tok.save_pretrained(args.output)
    example = tok(
        ["Rainwater enters the aquifer."],
        ["Water replenishes underground reserves."],
        return_tensors="pt",
    )
    with torch.inference_mode():
        torch.onnx.export(
            model,
            (example["input_ids"], example["attention_mask"]),
            str(args.output / "model.onnx"),
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            opset_version=17,
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
                "attention_mask": {0: "batch", 1: "sequence"},
                "logits": {0: "batch"},
            },
            dynamo=False,
        )
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(args.output / "model.onnx"),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    parity = []
    for n, length in [(1, 12), (4, 80), (1, 510)]:
        b = tok(
            ["source context " * length] * n,
            ["submitted sentence " * length] * n,
            padding=True,
            truncation="longest_first",
            max_length=512,
            return_tensors="pt",
        )
        with torch.inference_mode():
            expected = model(b["input_ids"], b["attention_mask"]).numpy()
        actual = session.run(
            None, {k: b[k].numpy() for k in ["input_ids", "attention_mask"]}
        )[0]
        np.testing.assert_allclose(actual, expected, atol=2e-4, rtol=2e-4)
        start = time.perf_counter()
        session.run(None, {k: b[k].numpy() for k in ["input_ids", "attention_mask"]})
        parity.append(
            {
                "batch": n,
                "tokens": b["input_ids"].shape[1],
                "max_absolute_error": float(np.max(np.abs(actual - expected))),
                "warm_seconds": time.perf_counter() - start,
            }
        )
    files = {f.name: sha(f) for f in args.output.iterdir() if f.is_file()}
    manifest = {
        "role": "review-reranker-v1",
        "architecture": "native-pair-roberta-cls-2logit",
        "checkpoint_sha256": WEIGHTS,
        "autonomous_classifier_accepted": False,
        "score_is_probability": False,
        "coverage_eligible": False,
        "license_notice": "Private noncommercial research use; ARC-Sim CC BY-NC-SA obligations apply. No commercial clearance.",
        "files": files,
        "parity": parity,
        "torch": torch.__version__,
        "onnxruntime": ort.__version__,
    }
    (args.output / "review-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
