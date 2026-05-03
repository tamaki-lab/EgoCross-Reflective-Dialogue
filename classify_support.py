"""
support set (train.json) の各問題を Gemini に question_type 分類させる。
結果を JSON で保存し、分布を表示する。
"""
import json
import os
import re
import time
from collections import Counter
from pathlib import Path

from google import genai
from google.genai import types
from tqdm import tqdm

BASE = Path(__file__).parent
TRAIN_JSON = BASE / "data/egocross/train.json"
OUTPUT_PATH = BASE / "outputs/support_question_types.json"

QUESTION_TYPES = [
    "action sequence identification",
    "action temporal localization",
    "animal identification",
    "dominant held-object identification",
    "interaction identification",
    "interaction temporal localization",
    "next action prediction",
    "next direction prediction",
    "next interaction prediction",
    "next phase prediction",
    "object counting",
    "object not visible identification",
    "object spatial localization",
    "special action identification",
    "sport identification",
]

CLASSIFY_PROMPT = """\
You are classifying questions from an egocentric video QA benchmark.

Classify the following question into exactly one of these question types:
{types_list}

Question: "{question}"

Reply with only the question type string, nothing else."""


def extract_question_text(content: str) -> str:
    return re.sub(r"<image>", "", content).strip().split("\nA.")[0].strip()


def match_question_type(raw: str) -> str:
    """完全一致 → 前方部分一致 の順でマッチ。"""
    raw = raw.strip().lower()
    # 完全一致 or 回答内に含まれる
    for qt in QUESTION_TYPES:
        if qt in raw:
            return qt
    # 前方部分一致（モデルが途中で切れた場合のフォールバック）
    for qt in QUESTION_TYPES:
        if qt.startswith(raw) and len(raw) >= 4:
            return qt
    return raw


def classify_question(client, question: str, model: str) -> str:
    prompt = CLASSIFY_PROMPT.format(
        types_list="\n".join(f"- {t}" for t in QUESTION_TYPES),
        question=question,
    )
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=1.0,       # gemini-3.x はthinking必須のため0.0不可
                    max_output_tokens=512,  # thinkingトークン消費後に回答が切れないよう余裕を持たせる
                    thinking_config=types.ThinkingConfig(thinking_budget=-1),
                ),
            )
            raw = response.text.strip().lower() if response.text else ""
            return match_question_type(raw)
        except Exception as e:
            wait = 2 ** attempt
            print(f"  Error attempt={attempt}: {e}  (retry in {wait}s)")
            time.sleep(wait)
    return "unknown"


def main():
    api_key = os.environ.get("GEMINI_API_KEY", "")
    vertex_key = os.environ.get("VERTEX_AI_API_KEY", "")

    if vertex_key:
        client = genai.Client(vertexai=True, api_key=vertex_key)
        model = "gemini-3.1-pro-preview"
        print("Using Vertex AI")
    elif api_key:
        client = genai.Client(api_key=api_key)
        model = "gemini-3.1-pro-preview"
        print("Using AI Studio")
    else:
        raise ValueError("GEMINI_API_KEY または VERTEX_AI_API_KEY が必要です")

    with open(TRAIN_JSON) as f:
        train_data = json.load(f)

    OUTPUT_PATH.parent.mkdir(exist_ok=True)

    results = []
    pbar = tqdm(train_data, desc="classifying")
    for i, item in enumerate(pbar):
        question = extract_question_text(item["messages"][0]["content"])
        predicted_type = classify_question(client, question, model)
        results.append({
            "index": i,
            "domain": item["domain"],
            "question": question,
            "predicted_type": predicted_type,
        })
        pbar.write(
            f"[{i:2d}] {item['domain']:10s}  {predicted_type:45s}  {question[:60]}")
        time.sleep(0.3)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nSaved -> {OUTPUT_PATH}")

    # 分布表示
    print("\n=== 分布 (domain × question_type) ===")
    by_domain = {}
    for r in results:
        by_domain.setdefault(r["domain"], []).append(r["predicted_type"])

    for domain, types_list in sorted(by_domain.items()):
        print(f"\n[{domain}]")
        for qt, n in Counter(types_list).most_common():
            print(f"  {qt:45s} {n}")

    unknown = [r for r in results if r["predicted_type"] not in QUESTION_TYPES]
    if unknown:
        print(f"\n⚠ 想定外の分類: {len(unknown)}件")
        for r in unknown:
            print(f"  [{r['index']}] '{r['predicted_type']}'")


if __name__ == "__main__":
    main()
