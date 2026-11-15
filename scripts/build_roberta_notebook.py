"""Build the manual free-GPU notebook and a private, data-free code bundle."""

from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import argparse, hashlib, json, ast

ROOT = Path(__file__).resolve().parents[1]


def cell(kind, source):
    value = {
        "cell_type": kind,
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }
    if kind == "code":
        value.update(execution_count=None, outputs=[])
        ast.parse(source)
    return value


cells = []


def md(s):
    cells.append(cell("markdown", s))


def code(s):
    cells.append(cell("code", s))


md("""# SemaTrace V3: source-linked RoBERTa fine-tuning
One bounded reproduction campaign of [Bär et al., 2021](https://aclanthology.org/2021.starsem-1.5/). Full RoBERTa encoder fine-tuning; actual ARC-Sim sentence labels. This is a reimplementation with documented choices, not an exact replication or trained plagiarism detector.

**Manual action:** select **Runtime → Change runtime type → GPU** (free runtime), then run cells in order. Upload `sematrace-roberta-code.zip` when prompted. Keep this notebook, Drive folder, data and resulting weights private: ARC-Sim is CC BY-NC-SA 3.0. Do not buy compute. No account credentials or API keys are requested.

Free GPU availability and session duration are variable. The training-only pilot reports measured throughput and an optimistic full-campaign estimate. Training saves every ten minutes and returns after three hours. If incomplete, reconnect to a free GPU and rerun setup, mount/upload and training cells with the **same code bundle**. The final cells refuse to run before all three epochs finish. This may require multiple sessions; no completion-time promise is made. See the bundled protocol for split hashes, baselines and release gates.
""")
code("""import sys, subprocess
assert (3,10) <= sys.version_info[:2] <= (3,12), "Use a Python 3.10–3.12 Colab runtime for the pinned wheels."
subprocess.run(["nvidia-smi"],check=True)
from google.colab import drive, files
from pathlib import Path
drive.mount('/content/drive')
ROOT=Path('/content/drive/MyDrive/SemaTrace-private/arc-roberta-v1')
ROOT.mkdir(parents=True,exist_ok=True)
CODE=Path('/content/sematrace-code')
CODE.mkdir(parents=True,exist_ok=True)
print('Private checkpoint/data folder:',ROOT)
""")
code("""from zipfile import ZipFile
import io, json, hashlib, shutil, os
uploaded=files.upload()
assert len(uploaded)==1 and next(iter(uploaded)).endswith('.zip'), 'Upload the single private code ZIP.'
payload=next(iter(uploaded.values()))
bundle_sha=hashlib.sha256(payload).hexdigest()
identity=ROOT/'code-bundle.sha256'
if identity.exists():
    assert identity.read_text().strip()==bundle_sha, 'Resume requires the original code ZIP; do not change a running campaign.'
else: identity.write_text(bundle_sha)
with ZipFile(io.BytesIO(payload)) as archive:
    for name in archive.namelist():
        assert not Path(name).is_absolute() and (CODE/name).resolve().is_relative_to(CODE.resolve()), 'Unsafe archive path'
    archive.extractall(CODE)
manifest=json.loads((CODE/'bundle-manifest.json').read_text())
for name,expected in manifest['files'].items():
    assert hashlib.sha256((CODE/name).read_bytes()).hexdigest()==expected, 'Bundle integrity failure: '+name
(ROOT/'inputs').mkdir(exist_ok=True)
shutil.copyfile(CODE/'inputs/legacy-head.npz',ROOT/'inputs/legacy-head.npz')
os.environ.update(HF_HOME=str(ROOT/'hf-cache'),TOKENIZERS_PARALLELISM='false')
print('Verified code bundle:',bundle_sha)
del uploaded,payload
""")
md("""## Pinned environment
Dependencies run in subprocesses so Colab's preimported NumPy/Torch modules cannot mix ABIs with this environment. No model code from the hub is executed (`trust_remote_code=False`). Downloads include the pinned MIT base checkpoint (~500 MB), public ARC archives and pinned MiniLM baseline. Keep at least 8 GB free in the private Drive folder for atomic checkpoints, optimizer state and data.
""")
code("""# A separate venv avoids Colab's preinstalled package conflicts; every command below uses it.
VENV=Path('/content/sematrace-venv')
if not (VENV/'bin/python').exists():
    subprocess.run([sys.executable,'-m','venv',str(VENV)],check=True)
PY=str(VENV/'bin/python')
subprocess.run([PY,'-m','pip','install','--only-binary=:all:','torch==2.5.1','--index-url','https://download.pytorch.org/whl/cu121'],check=True)
subprocess.run([PY,'-m','pip','install','--only-binary=:all:','-r',str(CODE/'backend/training/requirements.txt')],check=True)
def run(command):
    subprocess.run([PY,str(CODE/'backend/training/run.py'),command,'--root',str(ROOT)],check=True,cwd=CODE)
subprocess.run([PY,'-c',"import torch; assert torch.cuda.is_available(); print(torch.__version__,torch.cuda.get_device_name())"],check=True)
""")
md("""## Verify data and implementation
The complete official training split is retained. Validation/final sets conservatively exclude cross-split target papers, exact target sections and near-duplicate sources. Final also excludes all source families and target papers from the previously exposed 242-pair study. Preparation inspects identities and labels for integrity, not model outcomes. Tiny structural smoke weights are discarded from model selection.
""")
code("run('prepare')\nrun('status')\nrun('smoke')\n")
md("""## Training-only feasibility pilot
The pilot uses 256 labeled training sentences for eight optimizer steps, verifies gradients inside the transformer, checkpoint reload, finite inference, peak GPU memory and throughput. It does **not** train a replacement shallow head or measure validation/final quality. The pilot weights are discarded. Its time estimate is optimistic because length distribution, validation and checkpoint I/O differ.
""")
code("run('pilot')\nprint((ROOT/'pilot.json').read_text())\n")
md("""## Full pretrained encoder fine-tuning and automatic recovery
Adam 2e-5, weighted cross entropy (positive weight 5), effective batch 32 via microbatches of 2, three epochs, source sentences separated by `[SSS]`, one target sentence, native RoBERTa pair encoding, maximum 512 tokens with longest-first truncation. No surrounding target context. Validation F1 selects one epoch; final labels never select weights or thresholds.

The complete official training split contains **1,440,966 sentence examples per epoch**. Review the pilot estimate before committing time; do not pay for compute. A session returns after three hours. Rerun this cell until complete; it resumes the last atomic batch-boundary checkpoint. If the free quota is insufficient, return `pilot.json` and `progress.json` for an explicit resource decision; do not silently subsample or change training.
""")
code("""run('train')
if not (ROOT/'training-complete.json').exists():
    print((ROOT/'progress.json').read_text())
    raise RuntimeError('Session checkpoint saved. Resume this training cell in a free GPU session; do not run final evaluation yet.')
print((ROOT/'training-complete.json').read_text())
""")
md("""## Freeze validation selection, then open final once
Compare RoBERTa with TF-IDF SP, the unchanged pinned MiniLM encoder and the frozen previous supervised head on identical fresh partitions. Each method receives validation-only threshold selection. The previous head is **not retrained**. Research and automatic operating points are separate. Automatic release requires the predeclared precision/recall/negative-document FPR and baseline-gain gates. This does not establish source character boundaries or application transfer quality.

Do not edit code, tune thresholds, retrain or select another checkpoint after opening the final set. Interrupted final evaluation can resume only with the identical selection seal. Completed results are preserved.
""")
code("""if not (ROOT/'selection.json').exists(): run('calibrate')
print((ROOT/'selection.json').read_text())
run('final')
""")
md("""## CPU ONNX parity and private return artifact
Export verifies dynamic batch/sequence ONNX logits against PyTorch. A successful export does not enable production semantic coverage. The return ZIP contains selected weights/tokenizer, protocol, hashes, metrics and pilot/progress, but no ARC text or raw prediction rows. Keep weights private pending the license and application gates. Return this ZIP to the local project for CPU latency, attribution and domain-transfer evaluation.
""")
code("""run('export')
from zipfile import ZIP_DEFLATED
out=ROOT/'sematrace-roberta-return.zip'
names=['pilot.json','local-smoke.json','progress.json','prepared/manifest.json','training-complete.json','validation-history.json','best-selection.json','selection.json','test-opened.json','final-results.json','export.json','code-bundle.sha256']
with ZipFile(out,'w',ZIP_DEFLATED) as z:
    for name in names:
        path=ROOT/name
        if path.exists(): z.write(path,name)
    for path in (ROOT/'best').iterdir():
        if path.is_file():z.write(path,'best/'+path.name)
    z.write(CODE/'research/README.md','protocol.md')
print('Keep private:',out,'SHA256',hashlib.sha256(out.read_bytes()).hexdigest())
files.download(str(out))
""")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--legacy-head",
        type=Path,
        default=ROOT / ".local/benchmarks/arc-reuse-cpu-v1/reuse-head.npz",
    )
    args = parser.parse_args()
    if not args.legacy_head.is_file():
        parser.error(
            "Supply the authorized frozen baseline head with --legacy-head; it is deliberately not distributed. See research/README.md."
        )
    path = ROOT / "notebooks/sematrace_v3_roberta_training.ipynb"
    path.parent.mkdir(exist_ok=True)
    value = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
            "colab": {"name": path.name, "provenance": []},
            "accelerator": "GPU",
        },
        "cells": cells,
    }
    # Stable cell IDs make notebook validation/diffs reproducible.
    for i, c in enumerate(cells):
        c["id"] = f"sematrace-{i:02d}"
    path.write_text(
        json.dumps(value, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    files = list((ROOT / "backend/training").glob("*.py")) + [
        ROOT / "backend/training/requirements.txt"
    ]
    files += [
        ROOT / "configs" / name
        for name in ("roberta_training.yaml", "corpora-20261120.lock.json")
    ]
    files += [
        ROOT / "research/results" / name
        for name in ("arc-cpu-selection.json", "arc-cpu-results.json")
    ]
    files += [
        ROOT / "backend/corpus/minilm-onnx.lock.json",
        ROOT / "research/README.md",
    ]
    dest = ROOT / ".local/roberta-notebook"
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / "sematrace-roberta-code.zip"
    mapping = {str(p.relative_to(ROOT)).replace("\\", "/"): p for p in files}
    mapping["inputs/legacy-head.npz"] = args.legacy_head.resolve()
    manifest = {
        "private": True,
        "contains_arc_text": False,
        "files": {
            name: hashlib.sha256(p.read_bytes()).hexdigest()
            for name, p in mapping.items()
        },
    }
    with ZipFile(target, "w", ZIP_DEFLATED) as z:
        for name, p in mapping.items():
            z.write(p, name)
        z.writestr("bundle-manifest.json", json.dumps(manifest, indent=2))
    print(
        json.dumps(
            {
                "notebook": str(path),
                "private_bundle": str(target),
                "bundle_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                "cells": len(cells),
                "code_cells_compile": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
