#!/usr/bin/env node
import { createHash, randomInt, randomUUID } from "node:crypto";
import { readFile, stat, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { classifyTone } from "./rules.ts";
import {
  buildConversationAnswerInstructions,
} from "../../src/domain/conversation-answer-prompt.ts";
import { evidenceWorkflowPrompt } from "../../src/domain/workflow-prompt.ts";

const directory = path.dirname(fileURLToPath(import.meta.url));
const developmentCases = JSON.parse(await readFile(path.join(directory, "cases.json"), "utf8"));
const heldoutCases = JSON.parse(await readFile(path.join(directory, "cases.heldout.json"), "utf8"));
const contexts = JSON.parse(await readFile(path.join(directory, "contexts.json"), "utf8"));
const labels = ["GREETING", "IDENTITY", "CAPABILITIES", "EVERYDAY", "GOOGLE_RECOVERY", "SOFTWARE_GUIDE", "CLOUD_GPU_GUIDE", "COMPANY_POLICY", "GENERAL_GUIDE"];
const sentiments = ["NEGATIVE", "NEUTRAL", "POSITIVE"];
const schema = {
  type: "object",
  additionalProperties: false,
  properties: {
    label: { type: "string", enum: labels },
    text: { type: "string" },
    knowledgeIds: { type: "array", items: { type: "string" } },
    evidence: {
      type: "array",
      items: {
        type: "object",
        additionalProperties: false,
        properties: {
          knowledgeId: { type: "string" },
          quote: { type: "string" },
        },
        required: ["knowledgeId", "quote"],
      },
    },
  },
  required: ["label", "text", "knowledgeIds", "evidence"],
};
const variants = [
  { id: "current_prompt", promptVariant: "current_prompt", useRuleSignal: false },
  { id: "infer_tone_in_prompt", promptVariant: "infer_tone_in_prompt", useRuleSignal: false },
  { id: "prompt_plus_local_rules", promptVariant: "prompt_plus_local_rules", useRuleSignal: true },
];

function argument(name, fallback) {
  const index = process.argv.indexOf(name);
  return index === -1 ? fallback : process.argv[index + 1];
}

function validateCaseData(cases) {
  const ids = new Set();
  for (const item of cases) {
    if (ids.has(item.id)) throw new Error(`Duplicate test ID: ${item.id}`);
    ids.add(item.id);
    if (!sentiments.includes(item.sentiment)) throw new Error(`Invalid sentiment: ${item.id}`);
    const context = contexts[item.contextId];
    if (!context || context.label !== item.label) throw new Error(`Missing or mismatched context: ${item.id}`);
    if (context.answer.length < 12 || !context._id.startsWith("support-kb-v1-"))
      throw new Error(`Context is not a reviewed seed snapshot: ${item.id}`);
  }
}

function confusionMatrix(rows) {
  return Object.fromEntries(
    sentiments.map((expected) => [
      expected,
      Object.fromEntries(
        sentiments.map((predicted) => [
          predicted,
          rows.filter((row) => row.expected === expected && row.predicted === predicted).length,
        ]),
      ),
    ]),
  );
}

function ruleMetrics(cases) {
  const processBefore = process.memoryUsage();
  const rows = cases.map((item) => {
    const started = performance.now();
    const result = classifyTone(item.question);
    return {
      id: item.id,
      expected: item.sentiment,
      ...result,
      predicted: result.label,
      latencyMs: performance.now() - started,
    };
  });
  const processAfter = process.memoryUsage();
  const latencies = rows.map((row) => row.latencyMs).sort((a, b) => a - b);
  const percentile = (fraction) => latencies[Math.max(0, Math.ceil(latencies.length * fraction) - 1)];
  const matrix = confusionMatrix(rows);
  const perLabel = Object.fromEntries(sentiments.map((label) => {
    const truePositive = matrix[label][label];
    const predicted = sentiments.reduce((sum, expected) => sum + matrix[expected][label], 0);
    const actual = sentiments.reduce((sum, value) => sum + matrix[label][value], 0);
    const precision = predicted ? truePositive / predicted : 0;
    const recall = actual ? truePositive / actual : 0;
    return [label, { precision, recall, f1: precision + recall ? 2 * precision * recall / (precision + recall) : 0 }];
  }));
  return {
    accuracy: rows.filter((row) => row.expected === row.predicted).length / rows.length,
    macroF1: sentiments.reduce((sum, label) => sum + perLabel[label].f1, 0) / sentiments.length,
    coverage: rows.length / cases.length,
    confidence: "not_available_for_phrase_rules",
    confusionMatrix: matrix,
    perLabel,
    inference: {
      totalLatencyMs: Number(rows.reduce((sum, row) => sum + row.latencyMs, 0).toFixed(3)),
      p50LatencyMs: Number(percentile(0.5).toFixed(3)),
      p95LatencyMs: Number(percentile(0.95).toFixed(3)),
      classifierSourceBytes: null,
      modelArtifactBytes: 0,
      processHeapDeltaBytes: processAfter.heapUsed - processBefore.heapUsed,
      processRssDeltaBytes: processAfter.rss - processBefore.rss,
      memoryNote: "Single-process before/after snapshots; not peak or isolated classifier memory.",
    },
    predictions: rows.map((row) => ({
      id: row.id,
      expected: row.expected,
      label: row.label,
      positiveCues: row.positiveCues,
      negativeCues: row.negativeCues,
      predicted: row.predicted,
    })),
  };
}

function outputChecks(answer, item, context) {
  const shape = Boolean(
    answer &&
      labels.includes(answer.label) &&
      typeof answer.text === "string" &&
      answer.text.trim().length >= 12 &&
      Array.isArray(answer.knowledgeIds) &&
      Array.isArray(answer.evidence),
  );
  const evidenceRequired = ["GOOGLE_RECOVERY", "CLOUD_GPU_GUIDE", "COMPANY_POLICY"].includes(item.label);
  const evidence = shape &&
    (!evidenceRequired || answer.evidence.length > 0) &&
    answer.knowledgeIds.every((id) => id === context._id) &&
    answer.knowledgeIds.every((id) => answer.evidence.some((entry) =>
      entry.knowledgeId === id && entry.quote.length >= 12 && entry.quote.length <= 300 && context.answer.includes(entry.quote),
    )) &&
    answer.evidence.every((entry) =>
      entry.knowledgeId === context._id && entry.quote.length >= 12 && entry.quote.length <= 300 && context.answer.includes(entry.quote),
    ) &&
    (answer.text.match(/https?:\/\/[^\s<>)\]]+/g) ?? []).every((url) => context.sources.some((source) => source.url === url.replace(/[.,;]$/, "")));
  return {
    responseShapeValid: shape,
    knowledgeEvidenceValid: Boolean(evidence),
    expectedLabelMatched: shape && answer.label === item.label,
  };
}

async function requestAnswer(apiKey, model, item, variant) {
  const context = contexts[item.contextId];
  let instructions = `${evidenceWorkflowPrompt}\n\n${buildConversationAnswerInstructions(variant.promptVariant)}`;
  const article = {
    _id: context._id,
    label: context.label,
    title: context.title,
    answer: context.answer,
    sources: context.sources,
  };
  const data = {
    question: item.question,
    labels,
    articles: [article],
    publicWeb: "",
    sourceScope: context.sources.map(({ title, scope }) => ({ title, scope })),
  };
  const toneSignal = classifyTone(item.question);
  if (variant.useRuleSignal)
    data.responseStyle = toneSignal;
  const started = performance.now();
  const response = await fetch("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    signal: AbortSignal.timeout(12000),
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model,
      store: false,
      max_completion_tokens: 1000,
      messages: [
        { role: "system", content: instructions },
        { role: "user", content: JSON.stringify(data) },
      ],
      response_format: {
        type: "json_schema",
        json_schema: { name: "support_assistance", strict: true, schema },
      },
    }),
  });
  const latencyMs = Number((performance.now() - started).toFixed(1));
  if (!response.ok) {
    // Do not echo upstream response bodies: they can contain account metadata.
    const failure = new Error(`OpenAI request failed with HTTP ${response.status}`);
    failure.stopBenchmark = true;
    throw failure;
  }
  const payload = await response.json();
  const choice = payload.choices?.[0];
  if (!choice?.message?.content || choice.finish_reason !== "stop")
    throw new Error(`OpenAI returned no complete answer (${choice?.finish_reason ?? "empty"})`);
  let answer;
  try {
    answer = JSON.parse(choice.message.content);
  } catch {
    throw new Error("OpenAI returned invalid JSON");
  }
  return {
    answer,
    checks: outputChecks(answer, item, context),
    latencyMs,
    usage: payload.usage ?? null,
    toneSignal: variant.useRuleSignal ? toneSignal : undefined,
  };
}

const offline = process.argv.includes("--offline");
const set = argument("--set", "development");
if (!["development", "heldout"].includes(set)) throw new Error("--set must be development or heldout");
const cases = set === "heldout" ? heldoutCases : developmentCases;
validateCaseData(cases);
const caseFile = set === "heldout" ? "cases.heldout.json" : "cases.json";
const caseHash = createHash("sha256").update(await readFile(path.join(directory, caseFile))).digest("hex");
const limit = Number(argument("--limit", String(cases.length)));
if (!Number.isInteger(limit) || limit < 1) throw new Error("--limit must be a positive integer");
const selectedCases = cases.slice(0, Math.min(limit, cases.length));
const model = process.env.AI_CONVERSATION_MODEL || process.env.AI_MODEL || "";
const apiKey = process.env.OPENAI_API_KEY || "";
if (!offline && !apiKey) throw new Error("OPENAI_API_KEY is not configured; no API requests were sent. Use --offline to score the local rules only.");
if (!offline && !model) throw new Error("Set AI_CONVERSATION_MODEL or AI_MODEL; no API requests were sent.");
const classifierSourceBytes = (await stat(path.join(directory, "rules.ts"))).size;
const localRuleBaseline = ruleMetrics(selectedCases);
localRuleBaseline.inference.classifierSourceBytes = classifierSourceBytes;

const results = {
  createdAtUtc: new Date().toISOString(),
  dataset: {
    path: `benchmarks/answer-style/${caseFile}`,
    split: set,
    sha256: caseHash,
    count: selectedCases.length,
    expectedSentimentCounts: Object.fromEntries(sentiments.map((label) => [label, selectedCases.filter((item) => item.sentiment === label).length])),
    syntheticOnly: true,
  },
  setup: {
    offline,
    model: model || null,
    expectedApiCalls: offline ? 0 : selectedCases.length * variants.length,
    webSearchCalls: 0,
    apiStorage: "store=false",
    independentHumanStyleReview: "not performed",
    variantsSentInRandomOrder: true,
  },
  localRuleBaseline,
  variants: Object.fromEntries(variants.map((variant) => [variant.id, {
    status: offline ? "not_run_offline_mode" : "pending",
    callsCompleted: 0,
    answers: [],
  }])),
};

if (!offline) {
  const schedule = selectedCases.flatMap((item) =>
    variants
      .map((variant) => ({ item, variant, order: randomInt(0, 0x1_0000_0000) }))
      .sort((left, right) => left.order - right.order),
  );
  for (const { item, variant } of schedule) {
    const destination = results.variants[variant.id];
    destination.status = "completed";
    try {
      const run = await requestAnswer(apiKey, model, item, variant);
      destination.answers.push({
        reviewId: randomUUID(),
        caseId: item.id,
        expectedLabel: item.label,
        expectedSentiment: item.sentiment,
        scenario: item.scenario,
        question: item.question,
        answer: run.answer,
        checks: run.checks,
        latencyMs: run.latencyMs,
        usage: run.usage,
        ...(run.toneSignal ? { toneSignal: run.toneSignal } : {}),
      });
      destination.callsCompleted += 1;
      console.log(`${variant.id}: ${item.id} complete (${run.latencyMs} ms)`);
    } catch (error) {
      destination.status = "partial_or_failed";
      destination.answers.push({
        caseId: item.id,
        error: error instanceof Error ? error.message : "Unknown evaluation error",
      });
      if (error?.stopBenchmark) {
        for (const pending of schedule) {
          const unstarted = results.variants[pending.variant.id];
          if (unstarted.status === "pending")
            unstarted.status = "not_started_after_api_error";
        }
        break;
      }
    }
  }
}

for (const [variantId, result] of Object.entries(results.variants)) {
  const successful = result.answers.filter((item) => item.answer);
  if (!successful.length) continue;
  const latencies = successful.map((item) => item.latencyMs).sort((a, b) => a - b);
  const percentile = (fraction) => latencies[Math.max(0, Math.ceil(latencies.length * fraction) - 1)];
  result.summary = {
    expectedAnswers: selectedCases.length,
    answersReturned: successful.length,
    answerLabelAccuracy: successful.filter((item) => item.checks.expectedLabelMatched).length / successful.length,
    responseShapeValidRate: successful.filter((item) => item.checks.responseShapeValid).length / successful.length,
    knowledgeEvidenceValidRate: successful.filter((item) => item.checks.knowledgeEvidenceValid).length / successful.length,
    p50LatencyMs: percentile(0.5),
    p95LatencyMs: percentile(0.95),
    totalInputTokens: successful.reduce((sum, item) => sum + (item.usage?.prompt_tokens ?? 0), 0),
    totalOutputTokens: successful.reduce((sum, item) => sum + (item.usage?.completion_tokens ?? 0), 0),
    totalTokens: successful.reduce((sum, item) => sum + (item.usage?.total_tokens ?? 0), 0),
    humanStyleReview: "not performed",
  };
  if (result.callsCompleted < selectedCases.length && result.status === "completed")
    result.status = "incomplete";
  void variantId;
}

const baselineUsage = new Map(
  results.variants.current_prompt.answers
    .filter((item) => item.answer && item.usage)
    .map((item) => [item.caseId, item.usage]),
);
for (const result of Object.values(results.variants)) {
  if (!result.summary || result === results.variants.current_prompt) continue;
  const comparable = result.answers.filter(
    (item) => item.answer && item.usage && baselineUsage.has(item.caseId),
  );
  const inputDelta = comparable.reduce(
    (sum, item) => sum + item.usage.prompt_tokens - baselineUsage.get(item.caseId).prompt_tokens,
    0,
  );
  const outputDelta = comparable.reduce(
    (sum, item) => sum + item.usage.completion_tokens - baselineUsage.get(item.caseId).completion_tokens,
    0,
  );
  result.summary.comparedAnswersToBaseline = comparable.length;
  result.summary.inputTokenDeltaVsCurrentPrompt = inputDelta;
  result.summary.inputTokenDeltaPerAnswer = comparable.length
    ? Number((inputDelta / comparable.length).toFixed(2))
    : null;
  result.summary.outputTokenDeltaVsCurrentPrompt = outputDelta;
}

const outputPath = path.resolve(argument("--output", path.join(directory, "results.latest.json")));
await writeFile(outputPath, `${JSON.stringify(results, null, 2)}\n`, "utf8");
console.log(JSON.stringify({
  outputPath,
  cases: selectedCases.length,
  ruleAccuracy: Number(results.localRuleBaseline.accuracy.toFixed(4)),
  apiCallsExpected: results.setup.expectedApiCalls,
  apiCallsCompleted: Object.values(results.variants).reduce((sum, value) => sum + value.callsCompleted, 0),
  variantStatus: Object.fromEntries(Object.entries(results.variants).map(([id, value]) => [id, value.status])),
  status: offline ? "offline rule benchmark only" : "api benchmark finished",
}, null, 2));

if (!offline && Object.values(results.variants).some((variant) => ["partial_or_failed", "not_started_after_api_error", "incomplete"].includes(variant.status)))
  process.exitCode = 1;
