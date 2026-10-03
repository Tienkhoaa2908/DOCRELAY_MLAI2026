// The same transparent phrase baseline evaluated in benchmarks/sentiment,
// reused here as a response-style hint rather than as a routing decision.
const positivePhrases = [
  "cam on", "rat hai long", "hai long voi", "khong co gi de phan nan",
  "rat ro rang", "de hieu", "rat tot", "tuyet voi", "hoat dong tot",
  "xu ly tot", "tron tru", "rat huu ich", "huu ich", "than thien",
  "hieu qua", "chuyen nghiep", "deu chay", "cam thay yen tam", "toi vui",
  "rat thich", "appreciate", "thanks", "thank you", "worked perfectly",
  "helpful", "patient", "great job", "working again", "solved it",
  "not bad", "awesome", "excellent",
];

const negativePhrases = [
  "khong hai long", "khong tot", "khong huu ich", "that vong", "rat buc",
  "buc minh", "phat cau", "kho chiu", "te qua", "chan that", "rat lo",
  "qua met moi", "khong giai quyet", "phan hoi cham", "lien tuc bi chuyen",
  "treo suot", "frustrat", "not happy", "not helpful", "disappointed",
  "annoying", "annoyed", "useless", "terrible", "worse", "ignored my issue",
  "cannot believe", "can't believe", "khong the tin duoc", "cham that",
  "cho ca ngay", "dung gui cau tra loi tu dong nua", "phien phuc",
];

function normalize(text: string) {
  return text
    .toLowerCase()
    .replaceAll("đ", "d")
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/\s+/g, " ")
    .trim();
}

export function classifyTone(text: string) {
  const normalized = normalize(text);
  const positiveCues = positivePhrases.filter((phrase) => normalized.includes(phrase));
  const negativeCues = negativePhrases.filter((phrase) => normalized.includes(phrase));
  const label =
    negativeCues.length > positiveCues.length
      ? "NEGATIVE"
      : positiveCues.length > negativeCues.length
        ? "POSITIVE"
        : "NEUTRAL";
  return { label, positiveCues, negativeCues } as const;
}
