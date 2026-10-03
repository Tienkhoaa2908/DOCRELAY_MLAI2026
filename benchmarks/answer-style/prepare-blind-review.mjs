#!/usr/bin/env node
import { randomUUID } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const directory = path.dirname(fileURLToPath(import.meta.url));
const inputPath = process.argv[2];
if (!inputPath) throw new Error("Usage: node prepare-blind-review.mjs <api-results.json>");

const resultsPath = path.resolve(inputPath);
const results = JSON.parse(await readFile(resultsPath, "utf8"));
const casePath = path.join(directory, path.basename(results.dataset?.path ?? ""));
const cases = JSON.parse(await readFile(casePath, "utf8"));
const contexts = JSON.parse(await readFile(path.join(directory, "contexts.json"), "utf8"));
const caseById = new Map(cases.map((item) => [item.id, item]));
const blind = [];
const key = [];

for (const [variant, result] of Object.entries(results.variants ?? {})) {
  for (const item of result.answers ?? []) {
    if (!item.answer) continue;
    const testCase = caseById.get(item.caseId);
    const context = testCase ? contexts[testCase.contextId] : undefined;
    if (!testCase || !context) continue;
    const reviewId = randomUUID();
    blind.push({
      reviewId,
      question: testCase.question,
      reviewedKnowledge: {
        title: context.title,
        answer: context.answer,
        sources: context.sources,
      },
      response: {
        text: item.answer.text,
        knowledgeIds: item.answer.knowledgeIds,
        evidence: item.answer.evidence,
      },
      ratings: {
        factualGroundingAndEvidence: null,
        taskResolution: null,
        naturalness: null,
        empathyFit: null,
        directness: null,
        clarity: null,
        unnecessaryApology: null,
        safetyAndSchema: null,
        comments: "",
      },
    });
    key.push({ reviewId, caseId: item.caseId, variant });
  }
}

for (let index = blind.length - 1; index > 0; index -= 1) {
  const other = Math.floor(Math.random() * (index + 1));
  [blind[index], blind[other]] = [blind[other], blind[index]];
}

const base = resultsPath.replace(/\.json$/i, "");
const blindPath = `${base}.review-blind.json`;
const keyPath = `${base}.review-key.json`;
await writeFile(blindPath, `${JSON.stringify({ instructions: "Rate each response without trying to infer its variant. Keep reviewId unchanged when returning ratings.", responses: blind }, null, 2)}\n`, "utf8");
await writeFile(keyPath, `${JSON.stringify({ warning: "Keep this file separate from the blinded reviewer copy until ratings are complete.", assignments: key }, null, 2)}\n`, "utf8");
console.log(JSON.stringify({ blindedResponses: blind.length, blindPath, keyPath }));
