# Semantic reuse research

This directory records a completed, computationally constrained adaptation of [Recovering Lexically and Semantically Reused Texts](https://aclanthology.org/2021.starsem-1.5/). It preserves the useful results and failed acceptance decision. The experiment is separate from the application's optional OpenAI review.

## Model and task

A full RoBERTa-base encoder, approximately 124 million parameters, receives a cited abstract paired with one target sentence. Source sentences are separated by the added `[SSS]` token. The first-token representation passes through dropout 0.1 and a two-logit linear head. The encoder and head are both fine-tuned using weighted classification loss; maximum input length is512 tokens with native pair truncation.

The annotations come from the [official ARC-Sim partitions](https://github.com/maclaughlin/ARC-Sim). They operationalize citation-linked sentence reuse and are not universal labels for every kind of plagiarism. ARC-Sim's CC BY-NC-SA 3.0 terms apply; raw text, private heads and trained weights are not redistributed here.

## Fixed practical subset

The original full-data GPU pilot worked but projected 147.68 hours. A separate training-only subset made the experiment practical without changing the encoder architecture or silently reducing its sequence limit.

| Official-training stratum | Examples |
| --- | ---: |
| Positive, low lexical overlap | 1,500 |
| Other positive | 1,500 |
| Related negative sections, high overlap | 6,000 |
| Negative sentences inside positive sections, high overlap | 6,000 |
| Other related negative sections | 9,000 |
| Other negative sentences inside positive sections | 6,000 |
| **Total** |**30,000** |

Duplicate input pairs and conflicting-label keys were excluded. Deterministic source-family round robin and four length quartiles within each stratum represented 4,978 training source families. The source/target unique-word Jaccard≤0.10 definition identifies the low-overlap slice; it is not a production cosine threshold. Related citation-linked sections are a proxy for topical negatives, not a claim of equal manually verified difficulty.

The subset contains 3,000 positives and 27,000 negatives. Positive class weight changed from5 to **1.892368104**, preserving the original aggregate weighted class odds after the prevalence change:

```text
5 × (58,151 / 1,382,815) × (27,000 / 3,000)
```

Sampling used only official training. Validation/final identities, source-family separation, earlier-exposure exclusions and evaluation seals were preserved. The selected subset identity is locked in `configs/roberta_practical_selection.lock.json`.

## Training and recovery

The practical configuration retained three epochs, learning rate 2e-5 and effective batch 32. T4 execution used FP16/scaling, microbatch 8 with accumulation, dynamic padding to a multiple of 8, and deterministic length grouping. No encoder layers were frozen. AMP overflow retried the same batch rather than silently dropping examples.

A representative training-only throughput pilot preceded full execution. Private checkpoints retained optimizer, scaler, RNG and cursor state, with alternating integrity-checked slots and identity checks for resumption. Training/evaluation refuse incompatible or already-opened campaign state.

Actual results: **three epochs,2,814 optimizer updates,924.42 seconds of GPU update time**. Update time excludes setup, validation and checkpoint overhead. The practical pilot measured99.83 examples/second. These timings cannot isolate the contribution of each optimization or guarantee another machine's runtime.

The successful cloud environment used Python 3.13.16, PyTorch2.6.0+cu124, Transformers4.51.3 and tokenizers0.21.1. The research requirements record those direct versions; the CPU product has a separate, smaller environment. Original full-data configuration/notebook remain available alongside the practical adaptation.

## Independent evaluation

The final partition contained 8,024 sentences,299 positives and 494 source/section pairs. The selected epoch and thresholds came from validation. The final results are exposed and must not guide another round presented as an untouched final evaluation.

| Method | Precision | Recall | F1 | Low-overlap F1 | Ranking MAP |
| --- | ---: | ---: | ---: | ---: | ---: |
| TF-IDF | .190883 | .224080 | .206154 | .115607 | .444436 |
| Pinned MiniLM | .162861 | .357860 | .223849 | .174061 | .481030 |
| Earlier frozen supervised MiniLM | .206452 | .214047 | .210181 | .175610 | .479713 |
| Practical RoBERTa |**.354839** |**.478261** |**.407407** |**.353468** |**.659137** |

At the validation-selected research threshold 0.74, RoBERTa produced143 TP/260 FP/156 FN/7,465 TN. Its F1 gain was18.36 percentage points over MiniLM and 20.13 points over TF-IDF. This is an observed result for this protocol, not application-wide accuracy or a significance claim.

Related-negative sentence FPR was49/3,122=1.57%; negative-document FPR was36/226=15.93%, counting any positive sentence in a negative pair as a document false alarm. Sentence and document statistics are different units.

### Why automatic acceptance failed

The F1-gain requirement of+.05 over TF-IDF passed. However, no validation threshold in the fixed grid simultaneously met sentence precision≥.80, negative-document FPR≤.10 and at least 30positive sentence predictions. Therefore the automatic operating point remained null; the additional automatic-recall requirement could not rescue it.

The research point's final precision was35.48%. Lowering acceptance criteria after seeing the result would not establish trustworthy autonomous use. The failed decision, original weights and seals remain preserved privately; no retraining or final-set retuning was performed for publication.

### Why MiniLM remains the default

Research ranking supplied the correct source abstract and ranked target sentences. It did not test finding the right source among competing documents.

A separate frozen, purpose-written transfer diagnostic compared source passages: MiniLM Hit@1 was16/16; the unchanged RoBERTa rerank was14/16. No gain was demonstrated. The full controlled fixture and result are retained; they are not a representative natural-data accuracy estimate.

An optional private ONNX export passed numerical parity and supports small CPU ranking comparisons: up to 8 pairs,512 tokens, two compute threads, bounded caching. A logit margin supplies ordinal order, not a calibrated reuse probability. It is never used for autonomous classification or matching-word coverage. Default retrieval remains local MiniLM plus BM25.

## Reproducibility

Application setup requires neither these research inputs nor training. The notebooks preserve the executed workflow for authorized reproduction; publication did not execute another training campaign.

Relevant files:

- [`backend/training/`](../backend/training): original preparation, model, metrics, training and evaluation.
- [`backend/training_practical/`](../backend/training_practical): frozen sampling, AMP/batching, checkpoint identity and constrained execution.
- [`configs/roberta_training.yaml`](../configs/roberta_training.yaml) and [`configs/roberta_practical.yaml`](../configs/roberta_practical.yaml).
- [Original notebook](../notebooks/sematrace_v3_roberta_training.ipynb) and [practical notebook](../notebooks/sematrace_v3_roberta_practical_30k.ipynb).
- [`results/`](results): aggregate outcomes, frozen protocol/selection metadata and controlled ranking diagnostic; no corpus prose or model weights.

To rebuild a private portable notebook/code bundle, supply the authorized frozen legacy baseline head and, for the practical campaign, its original selected input-index manifest:

```sh
python scripts/build_roberta_notebook.py --legacy-head /private/reuse-head.npz
python scripts/build_practical_roberta_notebook.py --legacy-head /private/reuse-head.npz --selection /private/selection.json
```

Outputs go to ignored `.local/` directories. The builders include code/configuration/hash manifests, and only the explicitly supplied private research inputs. Never add their ZIPs or weights to Git. These inputs are deliberately not downloadable from the public repository; exact historical baseline reproduction requires access to the preserved authorized artifacts. A public clone is sufficient for the application, but is not a self-contained distribution of restricted research inputs.

For reproduction, the original notebook prepares verified official archives and prior-exposure exclusions; the practical notebook takes that campaign folder plus its fixed subset manifest. Use the documented Python 3.13/T4 environment, run preparation and the training-only pilot, inspect the resource gate, and continue only within the protocol. Do not rerun an already opened final or modify the frozen hashes. Existing completed campaigns must resume only with their original private bundle/configuration, not repackaged publication code with a different code identity.

Windows CRLF versus Linux LF is handled by a verified newline-only portability path. Original campaign files remain untouched; expected frozen CRLF copies are written to the new prepared directory without changing rows, labels or frozen hashes. The standalone patch and byte-level regression tests remain included.

## Interpretation and provenance

The new OpenAI review is a distinct product method with literal grounding, not evidence that the RoBERTa gate passed. Its frozen nine-case provider check and UI demonstrations establish specific behavior, not independent precision/recall. Model explanations remain advisory.

Complete original archives, predictions where available, checkpoints and exposure records remain private. Aggregate verification can check hashes/seals and reported arithmetic; the returned package did not include every raw baseline prediction needed for unrestricted per-example recomputation. The preserved assessment records that limitation.
