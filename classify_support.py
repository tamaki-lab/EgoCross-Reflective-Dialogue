"""
support set (train.json) の各問題を Gemini に question_type 分類させる。
結果を JSON で保存し、分布を表示する。
"""
import json
import os
import re
import time
from collections import Counter

from google import genai
from google.genai import types
from tqdm import tqdm

from common import SUPPORT_JSON, OUTPUT_DIR

TRAIN_JSON = SUPPORT_JSON
OUTPUT_PATH = OUTPUT_DIR / "support_question_types.json"

DOMAIN_VALID_TYPES: dict[str, list[str]] = {
    "animal": [
        "animal identification",
        "interaction identification",
        "interaction temporal localization",
    ],
    "industry": [
        "action temporal localization",
        "dominant held-object identification",
        "next interaction prediction",
        "object counting",
        "object not visible identification",
        "object spatial localization",
    ],
    "surgery": [
        "action temporal localization",
        "dominant held-object identification",
        "next action prediction",
        "next phase prediction",
        "object counting",
        "object not visible identification",
        "object spatial localization",
    ],
    "xsports": [
        "action sequence identification",
        "action temporal localization",
        "next direction prediction",
        "special action identification",
        "sport identification",
    ],
}

CLASSIFY_PROMPT = """\
You are classifying questions from an egocentric video QA benchmark.

Classify the following question (including its answer options) into exactly one of these question types:
{types_list}

Question: "{question}"

Reply with only the question type string, nothing else."""


def extract_question_text(content: str) -> str:
    """<image> タグを除去して質問文＋選択肢を返す。"""
    return re.sub(r"<image>", "", content).strip()


def match_question_type(raw: str, valid_types: list[str]) -> str:
    """完全一致 → 前方部分一致 の順でマッチ。"""
    raw = raw.strip().lower()
    for qt in valid_types:
        if qt in raw:
            return qt
    for qt in valid_types:
        if qt.startswith(raw) and len(raw) >= 4:
            return qt
    return raw


def classify_question(client, question: str, domain: str, model: str) -> str:
    valid_types = DOMAIN_VALID_TYPES.get(
        domain, list(DOMAIN_VALID_TYPES.keys()))
    prompt = CLASSIFY_PROMPT.format(
        types_list="\n".join(f"- {t}" for t in valid_types),
        question=question,
    )
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=1.0,
                    max_output_tokens=512,
                    thinking_config=types.ThinkingConfig(thinking_budget=-1),
                ),
            )
            raw = response.text.strip().lower() if response.text else ""
            return match_question_type(raw, valid_types)
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
        predicted_type = classify_question(
            client, question, item["domain"], model)
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

    all_valid = {t for ts in DOMAIN_VALID_TYPES.values() for t in ts}
    unknown = [r for r in results if r["predicted_type"] not in all_valid]
    if unknown:
        print(f"\n⚠ 想定外の分類: {len(unknown)}件")
        for r in unknown:
            print(f"  [{r['index']}] '{r['predicted_type']}'")


if __name__ == "__main__":
    main()
