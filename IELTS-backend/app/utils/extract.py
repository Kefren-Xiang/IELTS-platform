# app/utils/extract.py
import re
from typing import Tuple

# ---------- 常见关键词 ---------- #
PROMPT_MARKERS = [
    r"topic\s+is[:：]?\s*",
    r"题(?:目|干)[:：]?\s*",
    r"题为[:：]?\s*",
    r"请写(?:一篇)?(?:雅思)?作文[:：]?\s*",
    r"write\s+(?:an|a)\s+(?:ielts\s+)?essay\s+about\s+",
    r"on\s+the\s+topic\s+of\s+",
]

ESSAY_MARKERS = [
    r"here'?s\s+my\s+essay[:：]?",
    r"以下(?:是一篇)?(?:雅思)?作文[:：]?",
    r"(?:范文|正文)[:：]?",
    r"my\s+essay\s*：?",
]

# ---------- 阈值与正则 ---------- #
ESSAY_SENTENCE_THRESHOLD = 3  # 句子数 ≥3 认为像完整作文
QUOTE_RE = re.compile(r"[\"“”'‘’]([^\"””'‘’]{5,120})[\"“”'‘’]")  # 捕获引号里 5‑120 字

def _clean(txt: str) -> str:
    return re.sub(r"\s{2,}", " ", txt.strip().replace("\r", " "))

# ---------- 主函数 ---------- #
def extract_prompt_and_essay(text: str) -> Tuple[str, str]:
    """返回 (prompt, essay)。其中为空字符串代表该部分未提供。"""
    raw = text.strip()

    # 0️⃣ 双空行启发：多段落粘贴
    if "\n\n" in raw:
        parts = [p.strip() for p in raw.split("\n\n") if p.strip()]
        if len(parts) >= 2:
            # 第一段短 → 题目；其余合并 → 作文
            if len(parts[0].split()) <= 25:
                prompt = parts[0]
                essay = " ".join(parts[1:])
                return _clean(prompt), _clean(essay)

    # 1️⃣ 先抓 essay marker
    essay = ""
    for pat in ESSAY_MARKERS:
        m = re.search(pat, raw, flags=re.I)
        if m:
            essay = raw[m.end():].strip()
            break

    # 2️⃣ 没 marker：用句子阈值猜测正文
    if not essay:
        if len(re.split(r"[.!?。！？]", raw)) - 1 >= ESSAY_SENTENCE_THRESHOLD:
            essay = raw

    # 3️⃣ 抓 prompt marker
    prompt = ""
    for pat in PROMPT_MARKERS:
        m = re.search(pat, raw, flags=re.I)
        if m:
            rest = raw[m.end():]
            prompt = re.split(r"[.!?。！？\n]", rest, maxsplit=1)[0]
            break

    # 4️⃣ 引号捕获（仅当 prompt 仍为空且 essay 为空或较短）
    if not prompt:
        q = QUOTE_RE.search(raw)
        if q:
            prompt = q.group(1)

    # 5️⃣ fallback：若 essay 为空 → 把整句当 prompt
    if not prompt and not essay:
        prompt = raw

    return _clean(prompt), _clean(essay)
