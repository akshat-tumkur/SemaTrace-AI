"""Produce the new notebook/bundle without editing the original campaign."""

from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import argparse, ast, hashlib, json

ROOT = Path(__file__).resolve().parents[1]
cells = []


def md(text):
    cells.append(
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": text.splitlines(keepends=True),
        }
    )


def code(text):
    ast.parse(text)
    cells.append(
        {
            "cell_type": "code",
            "metadata": {},
            "source": text.splitlines(keepends=True),
            "outputs": [],
            "execution_count": None,
        }
    )


md("""# SemaTrace V3 — practical 30,000-example RoBERTa adaptation
Full pretrained encoder fine-tuning of the published source-abstract/target-sentence task. **Computationally constrained adaptation, not exact academic reproduction. No accuracy improvement is claimed.**

The successful T4 pilot projected 147.68 hours for the original full data. This fixed subset contains 3,000 positives (1,500 low-overlap) and 27,000 actual labeled negatives across 4,978 training source families. Source-balanced sampling covers four length quartiles within each stratum. Positive weight is **1.892368104**, adjusted from5 to preserve the original aggregate weighted class ratio after increasing positive prevalence. No synthetic labels, MiniLM retraining or NLI substitute.

**Use a free GPU runtime.** Run cells in order, upload the new private code ZIP, and inspect the representative pilot. Full training is blocked if its conservative training+validation+checkpoint estimate exceeds4 hours. That is an estimate, not a runtime guarantee. Do not buy compute or change data/epochs to force a pass. Keep Drive files, data and weights private (ARC-Sim CC BY-NC-SA).

The original full-data notebook/config/bundle remain recoverable. This notebook uses a separate campaign folder and preserves the original validation/final partitions and seals.
""")
code("""import sys,subprocess,json,hashlib,os,shutil,io
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
assert sys.version_info[:2]==(3,13), 'Use the reported working Python3.13 Colab runtime; do not build old Python wheels from source.'
subprocess.run(['nvidia-smi'],check=True)
from google.colab import drive,files
drive.mount('/content/drive')
BASE=Path('/content/drive/MyDrive/SemaTrace-private')
ROOT=BASE/'arc-roberta-practical-v1'
default=BASE/'arc-roberta-v1'
value=input('Original full-data campaign folder (Enter for '+str(default)+'): ').strip()
PRIOR=Path(value) if value else default
assert ROOT.resolve()!=PRIOR.resolve(), 'New and original campaigns must be separate.'
ROOT.mkdir(parents=True,exist_ok=True)
CODE=Path('/content/sematrace-practical-code');CODE.mkdir(exist_ok=True)
print('New private campaign:',ROOT)
""")
code("""uploaded=files.upload()
assert len(uploaded)==1 and next(iter(uploaded)).endswith('.zip'),'Upload sematrace-roberta-practical-code.zip'
payload=next(iter(uploaded.values()));checksum=hashlib.sha256(payload).hexdigest()
binding=ROOT/'code-bundle.sha256'
with ZipFile(io.BytesIO(payload)) as archive:
    incoming=json.loads(archive.read('bundle-manifest.json'))
    for name,expected in incoming['files'].items():
        assert hashlib.sha256(archive.read(name)).hexdigest()==expected,'Bundle integrity failure: '+name
    if binding.exists() and binding.read_text().strip()!=checksum:
        previous=binding.read_text().strip()
        blocked=['pilot.json','last.json','progress.json','training-complete.json','selection.json','test-opened.json','best','selected-epochs','checkpoints']
        assert previous==incoming.get('portability_upgrade_from_bundle_sha256') and not any((ROOT/n).exists() for n in blocked) and not (PRIOR/'test-opened.json').exists() and not (ROOT.parent/'sematrace-roberta-final-seal.json').exists(),'Resume requires identical ZIP; this portability upgrade is allowed only before model work.'
        backup=ROOT/'code-bundle-before-newline-fix.sha256'
        if not backup.exists():backup.write_text(previous)
        (ROOT/'portability-bundle-upgrade.json').write_text(json.dumps({'previous_bundle_sha256':previous,'corrected_bundle_sha256':checksum},indent=2))
    for name in archive.namelist():
        assert not Path(name).is_absolute() and (CODE/name).resolve().is_relative_to(CODE.resolve()),'Unsafe archive path'
    archive.extractall(CODE)
manifest=json.loads((CODE/'bundle-manifest.json').read_text())
for name,expected in manifest['files'].items():
    assert hashlib.sha256((CODE/name).read_bytes()).hexdigest()==expected,'Integrity failure: '+name
binding.write_text(checksum)
(ROOT/'inputs').mkdir(exist_ok=True)
shutil.copyfile(CODE/'inputs/legacy-head.npz',ROOT/'inputs/legacy-head.npz')
os.environ.update(HF_HOME=str(ROOT/'hf-cache'),TOKENIZERS_PARALLELISM='false')
del uploaded,payload
print('Verified frozen bundle:',checksum)
""")
md("""## Working Python3.13 / PyTorch2.6 CUDA12.4 environment
Pins match the user's successful pilot package output. All installations require binary wheels. A private venv and fresh subprocesses avoid mixed NumPy/Torch modules from the Colab kernel. No source builds, API keys, paid service or remote model code.
""")
code("""VENV=Path('/content/sematrace-practical-venv')
if not (VENV/'bin/python').exists():subprocess.run([sys.executable,'-m','venv',str(VENV)],check=True)
PY=str(VENV/'bin/python')
subprocess.run([PY,'-m','pip','install','--only-binary=:all:','torch==2.6.0','--index-url','https://download.pytorch.org/whl/cu124'],check=True)
subprocess.run([PY,'-m','pip','install','--only-binary=:all:','-r',str(CODE/'backend/training_practical/requirements.txt')],check=True)
def run(command,extra=None):
    subprocess.run([PY,str(CODE/'backend/training_practical/run.py'),command,'--root',str(ROOT)]+(extra or []),cwd=CODE,check=True)
check="import torch,sys,importlib.metadata as m,json; names=['torch','transformers','tokenizers','numpy','scikit-learn','scipy','PyYAML','onnx','onnxruntime','safetensors','huggingface-hub']; assert torch.cuda.is_available(); assert torch.__version__.startswith('2.6.0') and torch.version.cuda=='12.4'; import transformers,tokenizers,numpy,sklearn,scipy,yaml,onnx,onnxruntime,safetensors; print(json.dumps({'python':sys.version,'gpu':torch.cuda.get_device_name(),'packages':{n:m.version(n) for n in names}},indent=2))"
environment=subprocess.check_output([PY,'-c',check],text=True)
freeze=subprocess.check_output([PY,'-m','pip','freeze'],text=True)
if not (ROOT/'environment.json').exists():(ROOT/'environment.json').write_text(environment)
if not (ROOT/'pip-freeze.txt').exists():(ROOT/'pip-freeze.txt').write_text(freeze)
history=ROOT/'environments';history.mkdir(exist_ok=True)
session_hash=hashlib.sha256((environment+freeze).encode()).hexdigest()
(history/(session_hash+'.json')).write_text(environment)
(history/(session_hash+'.txt')).write_text(freeze)
print(environment)
""")
md("""## Frozen data and one fixed selection
Reuse the original prepared data if present; otherwise verify/download the official archives. The original folder is read-only to this workflow. Abort if its RoBERTa final seal exists. The older242-pair MiniLM exposure is already excluded by the frozen partition protocol.

The uploaded selection contains training row/sentence identities, not text. Verify every label/identity/hash; never resample it. Full validation10,634 sentences and final8,024 sentences remain unchanged. Pretokenization is an engineering cache, not model inference. Word-length quartiles guide sampling; the token audit reports actual BPE lengths/truncation by stratum. Maximum512/native longest-first truncation is unchanged.

Prepared source files are first verified against their original manifest. A verified LF/CRLF-only difference is staged as exact frozen CRLF bytes in the new campaign; the original campaign is never rewritten. Content/hash mismatches still fail.
""")
code("""run('prepare',['--original-root',str(PRIOR),'--selection',str(CODE/'inputs/practical-selection.json')])
run('status')
run('tokenize')
print((ROOT/'tokens/manifest.json').read_text())
if not (ROOT/'local-practical-smoke.json').exists():run('smoke')
""")
md("""## Representative training-only T4 pilot
Four warmup updates, then32 measured updates (1,024 examples), sampled proportionally from the frozen subset with all strata/length quartiles represented. Use FP16 autocast+GradScaler, microbatch8, effective batch32, no frozen encoder and no gradient checkpointing. Padding occurs independently per microbatch, rounded to8; deterministic length grouping reduces padding. Adam/weighted cross entropy/learning rate/three epochs stay explicit.

The pilot verifies internal encoder gradients and real optimizer/scaler/RNG checkpoint reload. It measures checkpoint writing and uses additional training examples to estimate validation inference time. No validation/final prediction scores are inspected. Pilot weights are discarded. The conservative gate includes p90 update time, full validation per epoch and checkpoint I/O; final baselines/export are extra.
""")
code("""if not (ROOT/'pilot.json').exists():run('pilot')
pilot=json.loads((ROOT/'pilot.json').read_text())
print(json.dumps(pilot,indent=2))
files.download(str(ROOT/'pilot.json'))
assert pilot['within_resource_budget'],'Pilot exceeds the resource gate. Stop and return pilot.json; do not silently reduce data or epochs.'
""")
md("""## Full subset fine-tuning and private recovery
The primary method fine-tunes all RoBERTa encoder layers. Each of30,000 selected examples appears once per epoch for3 epochs; no replacement sampling. Two alternating checkpoint slots plus a hashed pointer preserve the previous complete state until the next state commits. Optimizer, scaler, RNG, epoch/batch cursor and data/code/environment identities resume together.

The command returns after a three-hour session boundary. If incomplete, reconnect to a free GPU and rerun setup/upload with the **same ZIP/private folder**, then this cell. Do not rerun the original full-data campaign. An actual four-hour training-time cap preserves partial state rather than silently reduce epochs or claim success.
""")
code("""if not (ROOT/'training-complete.json').exists():run('train')
if not (ROOT/'training-complete.json').exists():
    print((ROOT/'progress.json').read_text())
    raise RuntimeError('Session checkpoint saved. Resume this training cell; final evaluation remains blocked.')
print((ROOT/'training-complete.json').read_text())
""")
md("""## Validation selection, unchanged frozen baselines and final once
Use full unchanged validation for epoch/threshold selection, then one sealed independent final evaluation. Baselines retain their original definitions: full-training TF-IDF SP, pinned MiniLM and exact frozen old supervised head. No head retraining or new cosine experiment. TF-IDF fitting is a CPU step and can take additional time/memory.

Metrics include precision/recall/F1/confusion counts, low-overlap performance, related-negative FPR, negative-document FPR, MAP/Acc@K and runtime. Original automatic operating-point and baseline-gain acceptance criteria stay unchanged. No threshold tuning or further training after the final seal. Interrupted final evaluation resumes only the identical seal; complete results remain preserved.
""")
code("""if not (ROOT/'selection.json').exists():run('calibrate')
print((ROOT/'selection.json').read_text())
run('final')
""")
md("""## Export only after acceptance; return private evidence either way
Passing ARC sentence acceptance does not validate source-side character boundaries, local CPU cost or application transfer. Only accepted results proceed to ONNX parity/export. Failed results are preserved without an automatic new campaign. Return the ZIP for actual laptop CPU benchmarking, semantic localization and application integration. No semantic coverage or production switch is enabled by this notebook.
""")
code("""result=json.loads((ROOT/'final-results.json').read_text())
if result['research_quality_gate_passed'] and not (ROOT/'export.json').exists():run('export')
elif result['research_quality_gate_passed']:print('Existing accepted export preserved.')
else:print('Acceptance failed. Preserve results and return evidence; no automatic retraining/export.')
out=ROOT/'sematrace-roberta-practical-return.zip'
names=['pilot.json','local-practical-smoke.json','environment.json','pip-freeze.txt','progress.json','training-complete.json','validation-history.json','best-selection.json','selection.json','test-opened.json','final-results.json','export.json','code-bundle.sha256','tokens/manifest.json','subset/selection.json','prepared/manifest.json']
with ZipFile(out,'w',ZIP_DEFLATED) as z:
    for name in names:
        path=ROOT/name
        if path.exists():z.write(path,name)
    for path in (ROOT/'best').iterdir():
        if path.is_file():z.write(path,'best/'+path.name)
    for path in (ROOT/'environments').iterdir():
        if path.is_file():z.write(path,'environments/'+path.name)
    z.write(CODE/'research/README.md','protocol.md')
h=hashlib.sha256()
with out.open('rb') as f:
    for part in iter(lambda:f.read(1024*1024),b''):h.update(part)
print('Private return:',out,'SHA256',h.hexdigest())
print('Optimizer recovery slots stay in Drive. No raw ARC text or prediction rows are packaged. If browser download is slow, use the printed Drive location.')
files.download(str(out))
""")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--legacy-head",
        type=Path,
        default=ROOT / ".local/benchmarks/arc-reuse-cpu-v1/reuse-head.npz",
    )
    parser.add_argument(
        "--selection",
        type=Path,
        default=ROOT / ".local/roberta-practical-v1/subset/selection.json",
    )
    args = parser.parse_args()
    if not args.legacy_head.is_file() or not args.selection.is_file():
        parser.error(
            "Supply the authorized frozen --legacy-head and --selection inputs. They are deliberately not distributed; see research/README.md."
        )
    notebook = ROOT / "notebooks/sematrace_v3_roberta_practical_30k.ipynb"
    for i, c in enumerate(cells):
        c["id"] = f"practical-{i:02d}"
    value = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
            "accelerator": "GPU",
            "colab": {"name": notebook.name, "provenance": []},
        },
        "cells": cells,
    }
    notebook.write_text(
        json.dumps(value, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    paths = (
        list((ROOT / "backend/training").glob("*.py"))
        + list((ROOT / "backend/training_practical").glob("*.py"))
        + [ROOT / "backend/training_practical/requirements.txt"]
    )
    paths += [
        ROOT / "configs" / name
        for name in (
            "roberta_practical.yaml",
            "roberta_practical_selection.lock.json",
            "roberta_training.yaml",
            "corpora-20261120.lock.json",
        )
    ]
    paths += [
        ROOT / "research/results" / name
        for name in (
            "arc-cpu-selection.json",
            "arc-cpu-results.json",
            "roberta-full-gpu-pilot.json",
            "roberta-practical-selection.json",
        )
    ]
    paths += [
        ROOT / "backend/corpus/minilm-onnx.lock.json",
        ROOT / "research/README.md",
    ]
    paths += [ROOT / "notebooks/colab_fix_prepared_newlines.py"]
    mapping = {p.relative_to(ROOT).as_posix(): p for p in paths}
    mapping["inputs/legacy-head.npz"] = args.legacy_head.resolve()
    mapping["inputs/practical-selection.json"] = args.selection.resolve()
    manifest = {
        "private": True,
        "contains_arc_text": False,
        "campaign": "arc-roberta-practical-30k-v1",
        "files": {
            n: hashlib.sha256(p.read_bytes()).hexdigest() for n, p in mapping.items()
        },
    }
    manifest["portability_upgrade_from_bundle_sha256"] = (
        "73fab3cc78754410557baee4cf80402a186ad024f87fd3d1e27f164a19a1c134"
    )
    dest = ROOT / ".local/roberta-practical-notebook"
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / "sematrace-roberta-practical-code.zip"
    with ZipFile(target, "w", ZIP_DEFLATED) as z:
        for name, path in mapping.items():
            z.write(path, name)
        z.writestr("bundle-manifest.json", json.dumps(manifest, indent=2))
    result = {
        "notebook": str(notebook),
        "private_bundle": str(target),
        "bundle_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "code_cells_compile": True,
        "cells": len(cells),
    }
    (dest / "bundle.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
