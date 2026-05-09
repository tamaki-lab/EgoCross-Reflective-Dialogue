"""
Step 2: question_type ごとに support set で warm-up 会話を構築する。
outputs/support_question_types.json (Step 1 の出力) を読み込み、
各 (domain, question_type) グループで順番に問題を解かせ、
正解・不正解のフィードバック＋反省を含む会話を構築して
outputs/warmup_conversations.json に保存する。
"""
import argparse
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path

from google import genai
from google.genai import types

BASE = Path(__file__).parent
TRAIN_JSON = BASE / "data/egocross/train.json"
CLASSIFY_JSON = BASE / "outputs/support_question_types.json"
DEFAULT_OUTPUT = BASE / "outputs/warmup_conversations_gemini.json"

DEFAULT_MODEL = "gemini-3.1-flash-image-preview"

MIME_MAP = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".png": "image/png", ".gif": "image/gif", ".webp": "image/webp",
}


def requires_thinking(model: str) -> bool:
    return model.startswith("gemini-3")


def load_image_part(path: str) -> types.Part:
    p = Path(path)
    mime = MIME_MAP.get(p.suffix.lower(), "image/jpeg")
    return types.Part.from_bytes(data=p.read_bytes(), mime_type=mime)


def extract_answer(text: str) -> str:
    for pat in (
        r"Final\s*Answer\s*[:：]\s*\**\s*\(?\s*([A-D])",
        r"answer\s+is\s*[:：]?\s*\**\s*\(?\s*([A-D])",
        r"correct\s+(?:option|choice)\s+is\s*[:：]?\s*\**\s*\(?\s*([A-D])",
    ):
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1).upper()
    matches = re.findall(r"\b([A-D])\b", text)
    return matches[-1] if matches else "A"


def make_config(model: str, thinking_budget: int) -> types.GenerateContentConfig:
    effective_budget = thinking_budget
    if effective_budget == 0 and requires_thinking(model):
        effective_budget = -1
    thinking_cfg = types.ThinkingConfig(
        thinking_budget=effective_budget) if effective_budget >= 0 else None
    max_tokens = 8192 if effective_budget != 0 else 512
    return types.GenerateContentConfig(
        temperature=1.0 if effective_budget != 0 else 0.0,
        max_output_tokens=max_tokens,
        thinking_config=thinking_cfg,
    )


def call_model(client, model: str, contents: list, config) -> str:
    for attempt in range(3):
        try:
            resp = client.models.generate_content(
                model=model, contents=contents, config=config)
            return resp.text.strip() if resp.text else ""
        except Exception as e:
            wait = 2 ** attempt
            print(f"    Error attempt={attempt}: {e}  (retry in {wait}s)")
            time.sleep(wait)
    return ""


def build_warmup_for_group(
    client,
    group_items: list[dict],
    model: str,
    config,
    rate_limit_sleep: float,
    explain_correct: bool = False,
) -> tuple[list[dict], int, int]:
    """
    1グループ分の warm-up 会話を構築する。

    Returns:
        turns: JSON保存用のターンリスト (role, text, images フィールド)
        n_correct: 正解数
        n_total: 問題数
    """
    contents: list[types.Content] = []
    turns: list[dict] = []
    n_correct = 0

    for item in group_items:
        # User turn: 画像 + 問題文
        parts = [load_image_part(p) for p in item["images"]]
        parts.append(types.Part.from_text(text=item["prompt"]))
        contents.append(types.Content(role="user", parts=parts))
        turns.append(
            {"role": "user", "text": item["prompt"], "images": item["images"]})

        # Model の回答
        model_text = call_model(client, model, contents, config)
        if not model_text:
            model_text = "Final Answer: A"
        model_answer = extract_answer(model_text)
        contents.append(types.Content(role="model", parts=[
                        types.Part.from_text(text=model_text)]))
        turns.append({"role": "model", "text": model_text})

        gt = item["ground_truth"]
        correct = (model_answer == gt)
        if correct:
            n_correct += 1

        # フィードバック
        if correct:
            if explain_correct:
                feedback = (
                    f"Correct. Please briefly explain why {gt} is the right answer "
                    "based on what you observed in the frames."
                )
            else:
                feedback = "Correct."
        else:
            feedback = (
                f"The correct answer is {gt}. "
                "Please reflect on what you may have missed or misinterpreted in the frames."
            )
        contents.append(types.Content(role="user", parts=[
                        types.Part.from_text(text=feedback)]))
        turns.append({"role": "user", "text": feedback, "images": []})

        status = "✓" if correct else f"✗ pred={model_answer} gt={gt}"
        print(f"    {status}  {item['prompt'][:70]}")

        # 不正解は反省、正解かつ explain_correct は根拠説明を取得
        if not correct or explain_correct:
            response = call_model(client, model, contents, config)
            if response:
                contents.append(types.Content(role="model", parts=[
                                types.Part.from_text(text=response)]))
                turns.append({"role": "model", "text": response})
            time.sleep(rate_limit_sleep)

        time.sleep(rate_limit_sleep)

    return turns, n_correct, len(group_items)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Gemini モデル名 (default: {DEFAULT_MODEL})")
    parser.add_argument("--thinking-budget", type=int, default=0,
                        help="thinking budget: 0=無効/自動, -1=動的, N=最大N token (default: 0)")
    parser.add_argument("--rate-limit-sleep", type=float, default=1.0,
                        help="リクエスト間の sleep 秒数 (default: 1.0)")
    parser.add_argument("--use-vertex", action="store_true",
                        help="Vertex AI を使用")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT),
                        help=f"出力 JSON パス (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--explain-correct", action="store_true",
                        help="正解時にも根拠説明をモデルに生成させる (default: off)")
    args = parser.parse_args()

    # Client setup
    if args.use_vertex:
        key = os.environ.get("VERTEX_AI_API_KEY", "")
        if not key:
            raise ValueError("VERTEX_AI_API_KEY 環境変数が必要です")
        client = genai.Client(vertexai=True, api_key=key)
        print("Using Vertex AI")
    else:
        key = os.environ.get("GEMINI_API_KEY", "")
        if not key:
            raise ValueError("GEMINI_API_KEY 環境変数が必要です")
        client = genai.Client(api_key=key)
        print("Using AI Studio")

    # Step 1 の分類結果を読み込む
    if not CLASSIFY_JSON.exists():
        raise FileNotFoundError(
            f"{CLASSIFY_JSON} が見つかりません。先に classify_support.py を実行してください。")
    with open(CLASSIFY_JSON) as f:
        classify_results = json.load(f)

    with open(TRAIN_JSON) as f:
        train_data = json.load(f)

    assert len(classify_results) == len(
        train_data), "classify と train のサイズが一致しません"

    # train.json と分類結果をマージ
    merged = []
    for cr, td in zip(classify_results, train_data):
        prompt = re.sub(r"<image>", "", td["messages"][0]["content"]).strip()
        merged.append({
            "index": cr["index"],
            "domain": cr["domain"],
            "question_type": cr["predicted_type"],
            "images": td["images"],
            "prompt": prompt,
            "ground_truth": td["messages"][-1]["content"].strip().upper(),
        })

    # (domain, question_type) でグループ化
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for item in merged:
        groups[(item["domain"], item["question_type"])].append(item)

    print(f"\n{len(groups)} グループ:")
    for (domain, qt), items in sorted(groups.items()):
        print(f"  [{domain:10s}] {qt:45s} {len(items)}問")

    config = make_config(args.model, args.thinking_budget)
    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True)

    warmup_data = {}
    grand_correct = grand_total = 0

    for (domain, qt), group_items in sorted(groups.items()):
        key = f"{domain}::{qt}"
        print(f"\n=== {key} ({len(group_items)}問) ===")
        turns, n_correct, n_total = build_warmup_for_group(
            client, group_items, args.model, config, args.rate_limit_sleep,
            explain_correct=args.explain_correct)
        warmup_data[key] = {
            "domain": domain,
            "question_type": qt,
            "n_items": n_total,
            "n_correct": n_correct,
            "turns": turns,
        }
        grand_correct += n_correct
        grand_total += n_total
        print(f"  → {n_correct}/{n_total} correct ({n_correct/n_total*100:.0f}%)")

        # 途中経過を随時保存
        with open(output_path, "w") as f:
            json.dump(warmup_data, f, ensure_ascii=False, indent=2)

    print(f"\n=== 総合 ===")
    print(
        f"  Warm-up accuracy: {grand_correct}/{grand_total} = {grand_correct/grand_total*100:.1f}%")
    print(f"  Saved → {output_path}")


if __name__ == "__main__":
    main()
