# Sentiment-aware answer style evaluation

The production candidate is a short instruction appended to the existing answer prompt. The same Chat Completions request still generates the answer, so production adds no classifier runtime or additional API request. The prompt treats tone as wording guidance only. It does not supply a sentiment field, persist a label, or alter evidence, safety checks, routing, urgency, priority, authority, or policy.

`src/domain/conversation-answer-prompt.ts` owns the baseline prompt, style instruction, and prompt composer used by both production and this benchmark. The candidate variants are:

1. `current_prompt`: current baseline instructions.
2. `infer_tone_in_prompt`: baseline plus the style instruction; this is the production candidate.
3. `prompt_plus_local_rules`: candidate 2 plus a local phrase-rule signal. This remains benchmark-only because errors can misread quotations, mixed sentiment, and indirect language.

## Datasets

- `cases.json` is the 18-case development set (6 per sentiment).
- `cases.heldout.json` is a separate 24-case synthetic evaluation set (8 per sentiment). It includes Vietnamese with and without diacritics, English/code-switching, neutral errors and deadlines, confusion, indirect frustration, appreciation, mixed tone, negation, and emotional words inside quotations.
- `contexts.json` snapshots reviewed knowledge seed text. Tests check that each snapshot still matches the application seed and that all questions still pass the existing conversation router.

Both sets are synthetic. The held-out cases are hand-authored rather than sampled from independently labeled support tickets, so they measure these examples and failure patterns, not production accuracy. Do not change the phrase-rule dictionary after looking at held-out scores and then report those scores as independent evaluation.

## Offline classifier evaluation

This sends no network requests and evaluates only the local phrase-rule signal. The report includes a confusion matrix, per-class precision/recall/F1, accuracy, macro-F1, case-level errors, inference latency, source bytes, and process memory snapshots. Memory deltas are process-level before/after samples, not an isolated peak-memory measurement.

```powershell
node --experimental-strip-types benchmarks/answer-style/evaluate.mjs --offline --set development --output benchmarks/answer-style/results.development.json
node --experimental-strip-types benchmarks/answer-style/evaluate.mjs --offline --set heldout --output benchmarks/answer-style/results.heldout.json
node node_modules/vitest/vitest.mjs run tests/answer-style-benchmark.test.ts
```

The existing 60-case sentiment experiment is historical corroboration, not a held-out set for this experiment. Its phrase rules scored 73.3% accuracy and 0.737 macro-F1; PhoBERT scored 58.3% / 0.560 and ViSoBERT 56.7% / 0.536. In that experiment, PhoBERT int8 occupied 283,829,288 bytes and ViSoBERT int8 occupied 134,102,054 bytes. The current prompt-only approach has no model artifact. PhoBERT's model card declares MIT and describes sentiment fine-tuning on e-commerce reviews; that domain does not establish helpdesk performance. The pinned ViSoBERT model card does not declare a license, so deployment rights remain unclear. See the [PhoBERT card](https://huggingface.co/wonrax/phobert-base-vietnamese-sentiment) and [ViSoBERT card](https://huggingface.co/5CD-AI/Vietnamese-Sentiment-visobert).

Vercel now offers an up-to-5-GB Large Functions path for eligible projects using Fluid Compute; standard functions remain subject to the 250-MB uncompressed limit. Project eligibility and settings must be checked before relying on Large Functions. Even where deployable, the two evaluated checkpoints were less accurate on the authored helpdesk set and add cold-start, RAM, and operational costs. See [Vercel Function limits](https://vercel.com/docs/functions/limitations) and [Large Functions](https://vercel.com/changelog/vercel-functions-can-now-be-up-to-5-gb-in-package-size-7yAwSyCig0IQDXUIDistv/eadf06d6c3).

## OpenAI answer comparison

The API benchmark sends only synthetic user text and the same reviewed context to all variants. It uses one selected model, strict JSON schema, the production 1,000-token completion cap, a 12-second timeout, `store: false`, and no web search. Requests for the three variants are interleaved in randomized order. A `--limit 1` smoke run makes 3 requests; the full held-out comparison makes 72. The report records completed requests, actual input/output/total tokens, response latency, schema/evidence/label checks, and variant status. Token and dollar deltas are not estimated if the API does not return usage or if the selected model/pricing is unknown.

Only use a key already available in the approved secret environment; the runner reads process environment variables and never prints or stores the key. First run a one-case smoke test:

```powershell
$env:AI_CONVERSATION_MODEL = '<same model configured for conversation answers>'
node --experimental-strip-types benchmarks/answer-style/evaluate.mjs --set heldout --limit 1 --output benchmarks/answer-style/results.api-smoke.json
```

If the smoke run returns a complete schema-valid answer and expected usage, run the full held-out comparison:

```powershell
node --experimental-strip-types benchmarks/answer-style/evaluate.mjs --set heldout --output benchmarks/answer-style/results.api-heldout.json
```

The generating model is not a style grader. After an API run, prepare a randomly ordered blind review copy and a separate unblinding key:

```powershell
node --experimental-strip-types benchmarks/answer-style/prepare-blind-review.mjs benchmarks/answer-style/results.api-heldout.json
```

Share only the `*.review-blind.json` file with the reviewer. They should rate factual grounding/evidence, task resolution, naturalness, empathy fit, directness, clarity, unnecessary apologies, and safety/schema compliance. Keep the `*.review-key.json` file separate until ratings are complete. The runner's mechanical checks cover only a subset of grounding and schema behavior. No preference score should be reported until a separate human review is recorded.

The prompt-only style instruction adds characters to the existing system prompt and can increase input tokens on uncached model requests. It adds no model call and no completion tokens by itself. Measure actual input tokens for each variant using the API `usage` fields; the application model and API key were not available for the offline run that generated the checked-in reports. OpenAI's API documentation describes `store` for Chat Completions; this benchmark explicitly sends `store: false`. See [OpenAI data controls](https://platform.openai.com/docs/models/default-usage-policies-by-endpoint).
