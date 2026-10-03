import { createHash } from "node:crypto";
import { describe, expect, it } from "vitest";
import cases from "../benchmarks/answer-style/cases.json";
import heldoutCases from "../benchmarks/answer-style/cases.heldout.json";
import contexts from "../benchmarks/answer-style/contexts.json";
import { classifyTone } from "../benchmarks/answer-style/rules";
import {
  buildConversationAnswerInstructions,
  conversationAnswerInstructions,
  responseStyleInstructions,
} from "@/domain/conversation-answer-prompt";
import { conversationRoute } from "@/domain/conversation";
import { knowledgeSeed } from "@/domain/knowledge";
import { extractIntake } from "@/domain/text";
import type { SupportInput } from "@/domain/contracts";

describe("answer-style benchmark corpus", () => {
  it("contains balanced synthetic cases for the support assistant", () => {
    expect(cases).toHaveLength(18);
    expect(cases.map(({ id }) => id).filter((id, index, all) => all.indexOf(id) === index)).toHaveLength(18);
    expect(Object.fromEntries(["NEGATIVE", "NEUTRAL", "POSITIVE"].map((label) => [
      label,
      cases.filter((item) => item.sentiment === label).length,
    ]))).toEqual({ NEGATIVE: 6, NEUTRAL: 6, POSITIVE: 6 });
  });

  it("keeps a balanced held-out set separate from development examples", () => {
    const developmentIds = new Set(cases.map(({ id }) => id));
    expect(heldoutCases).toHaveLength(24);
    expect(
      heldoutCases.filter(({ id }) => developmentIds.has(id)),
    ).toHaveLength(0);
    expect(
      heldoutCases.map(({ id }) => id).filter((id, index, all) => all.indexOf(id) === index),
    ).toHaveLength(heldoutCases.length);
    expect(Object.fromEntries(["NEGATIVE", "NEUTRAL", "POSITIVE"].map((label) => [
      label,
      heldoutCases.filter((item) => item.sentiment === label).length,
    ]))).toEqual({ NEGATIVE: 8, NEUTRAL: 8, POSITIVE: 8 });
  });

  it("uses exact reviewed knowledge seed text for every supplied context", () => {
    for (const context of Object.values(contexts)) {
      const seed = knowledgeSeed.find((item) => item._id === context._id);
      expect(seed, context._id).toBeDefined();
      expect(context.label).toBe(seed?.label);
      expect(context.title).toBe(seed?.title);
      expect(context.answer).toBe(seed?.answer);
      expect(context.sources).toEqual(seed?.sources);
    }
  });

  it("routes every question through the application's chat intake", () => {
    for (const item of [...cases, ...heldoutCases]) {
      const input: SupportInput = {
        mode: "freeform",
        serviceGroup: "OTHER",
        rawText: item.question,
        fields: {},
        confirmed: true,
        idempotencyKey: `style-${item.id}`,
      };
      const request = extractIntake(input, []);
      const routed = conversationRoute(input, request);
      expect(
        routed?.label,
        `${item.id}: ${item.question}; intent=${request.intentLabel}; risks=${request.riskSignals.join(",")}; kind=${request.requestKind}`,
      ).toBe(item.label);
    }
  });
});

describe("answer-style candidates", () => {
  it("preserves the current production instructions as the baseline", () => {
    expect(conversationAnswerInstructions).toContain("answer it specifically, naturally and thoroughly");
    expect(conversationAnswerInstructions).toContain("Only use facts in retrieved context for organization-specific claims");
    expect(createHash("sha256").update(conversationAnswerInstructions).digest("hex")).toBe(
      "a59220b11b3250be9c113d8409255bdc973713ad33cd14767762675d30105b61",
    );
  });

  it("does not treat a technical error alone as negative sentiment", () => {
    expect(classifyTone("VPN trên Windows báo lỗi 809. Mình nên ghi lại thông tin nào?").label).toBe("NEUTRAL");
  });

  it("detects clear frustration and appreciation from realistic messages", () => {
    expect(classifyTone("Thật nản, tôi đã thử bốn lần mà vẫn không giải quyết được.").label).toBe("NEGATIVE");
    expect(classifyTone("Cảm ơn, mình đã tìm được trang khôi phục.").label).toBe("POSITIVE");
  });

  it("keeps the style prompt focused on delivery and the existing safety boundary", () => {
    expect(responseStyleInstructions).toContain("A technical problem or deadline alone does not imply frustration or urgency");
    expect(responseStyleInstructions).toContain("Do not mention tone labels");
    expect(responseStyleInstructions).toContain("Tone must not change facts, evidence, uncertainty");
  });

  it("uses the same prompt composer for production and benchmark variants", () => {
    expect(buildConversationAnswerInstructions()).toBe(
      `${conversationAnswerInstructions}\n\n${responseStyleInstructions}`,
    );
    expect(buildConversationAnswerInstructions("current_prompt")).toBe(
      conversationAnswerInstructions,
    );
    expect(buildConversationAnswerInstructions("prompt_plus_local_rules")).toContain(
      "responseStyle field is a low-cost heuristic",
    );
  });
});
