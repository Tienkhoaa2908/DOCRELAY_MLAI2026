# Sentiment benchmark (offline inference)

This experiment compares three sentiment approaches on the same 60 hand-authored, synthetic helpdesk messages:

1. A small Vietnamese/English phrase baseline in `rules.py`.
2. The PhoBERT sentiment checkpoint, with Vietnamese word segmentation as required by its model card.
3. The ViSoBERT sentiment checkpoint.

Both transformer checkpoints are also exported to dynamic int8 ONNX and evaluated with the ONNX Runtime CPU provider. The program downloads weights from Hugging Face, then runs inference locally; it does not call a hosted inference API. Network access is needed only to fetch the pinned model files and Python packages.

## Results from this checkout

The full run completed on Windows 11, Python 3.13.16, and a 12-logical-CPU machine. Latencies below are per message after model load; file sizes are decimal byte counts. Full precision means the downloaded PyTorch checkpoint, while int8 means the exported ONNX graph.

| Method | Accuracy | Macro-F1 | Negative precision / recall | p50 / p95 latency | Model bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| Phrase rules | 73.3% | 0.737 | 1.00 / 0.65 | 0.02 / 0.03 ms | included in source |
| PhoBERT | 58.3% | 0.560 | 0.60 / 0.60 | 37.95 / 48.77 ms | 540,069,101 |
| PhoBERT ONNX int8 | 58.3% | 0.561 | 0.59 / 0.50 | 9.32 / 12.06 ms | 283,829,288 |
| ViSoBERT | 56.7% | 0.536 | 0.52 / 0.65 | 39.09 / 48.23 ms | 390,297,116 |
| ViSoBERT ONNX int8 | 55.0% | 0.535 | 0.52 / 0.65 | 8.98 / 11.22 ms | 134,102,054 |

The rule baseline scored best on this authored set. PhoBERT int8 kept the same accuracy as its full checkpoint and cut measured p50 latency by about 75%; ViSoBERT int8 lost 1.7 percentage points of accuracy and also ran faster. These results do not establish that the rules or either model will work better on real tickets. The full confusion matrices, per-case predictions, errors, first-inference and load timings, and exact pinned revisions are in `results.latest.json`.

The phrase baseline here is a separate candidate for ticket tone. It does not execute the application's existing `src/domain/feedback.ts` classifier, which handles replies after a support answer; that classifier and its routing decisions are outside this benchmark.

## Run on Windows

From the repository root, run:

```powershell
.
benchmarks\sentiment\run.ps1
```

The script creates a Python 3.13 environment under `%LOCALAPPDATA%`, installs the fully pinned `requirements.lock.txt`, runs the unit tests, downloads the two model checkpoints, exports/quantizes ONNX and writes `benchmarks/sentiment/results.latest.json`. Model caches and exported weights live under the system temporary directory; the repository stores no model binaries. First run downloads about 1 GB of checkpoint files.

For the lightweight rule-only pass:

```powershell
$python = Join-Path $env:LOCALAPPDATA 'MLAI2026\sentiment-benchmark\Scripts\python.exe'
& $python benchmarks\sentiment\benchmark.py --rules-only
```

Use `--skip-onnx` to compare the Python checkpoints without ONNX export. This flag is for troubleshooting; the full run evaluates all methods.

## What the report measures

Every method sees the same balanced cases and three labels: `NEGATIVE`, `NEUTRAL`, and `POSITIVE`. The report includes accuracy, macro-F1, negative-label precision/recall, confusion matrices, misclassified case IDs, per-ticket p50/p95 latency, model-load time and checkpoint/quantized graph size.

The corpus is synthetic and was authored for this exploratory comparison; it is not independent or representative production ground truth. Its scores only compare the candidates on these 60 cases. They must not be described as expected real-world accuracy. Have IT reviewers label a consented/anonymized held-out ticket sample before selecting a model.

The rule baseline is deliberately small and explainable. It uses explicit phrases and does not infer that a technical error is negative emotion. This makes it cheap to run but weak on implied frustration, paraphrases, sarcasm and code-switching.

## Deployment boundaries

- Sentiment is a soft tone cue for reviewers. It must not change deterministic risk, urgency, priority, approval, escalation or closure decisions.
- `NEGATIVE` is not equivalent to urgent or unsafe. Low certainty or processing failure should become `NEUTRAL`/`UNKNOWN`.
- PhoBERT's card declares MIT; it was trained on e-commerce reviews and requires pre-segmented text. Support-ticket performance needs its own evaluation.
- ViSoBERT is evaluated here because it is a Vietnamese candidate, but its pinned model card does not declare a license. Do not ship its weights until the owner clarifies deployment rights.
- ONNX Runtime has a Node.js binding, so a Node server route is a possible local-inference path. The measured graph alone does not prove the real Next.js/Vercel function bundle, memory use, cold start, or throughput will fit; trace and measure the built deployment artifact on the target runtime. Vercel's standard function bundle limit is 250 MB uncompressed; its 5 GB large-function path requires eligible Fluid Compute configuration. PhoBERT int8 is already about 270.6 MiB before runtime and application files, while ViSoBERT int8 is about 127.9 MiB before those additions.

Model and runtime references: [PhoBERT sentiment model card](https://huggingface.co/wonrax/phobert-base-vietnamese-sentiment), [ViSoBERT model card](https://huggingface.co/5CD-AI/Vietnamese-Sentiment-visobert), [ONNX Runtime Node.js binding](https://onnxruntime.ai/docs/get-started/with-javascript/node.html), and [Vercel Function limits](https://vercel.com/docs/functions/limitations).

The `phenomenon` tags identify case patterns such as technical-neutral, negation, code-switching and implied outcome. They are descriptive only; the script reports aggregate metrics and leaves the original labels unchanged.
