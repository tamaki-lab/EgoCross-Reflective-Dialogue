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

DOMAIN_ORIG_FPS = {"surgery": 25.0, "industry": 30.0, "xsports": 30.0, "animal": 30.0}


def _compute_eval_timestamps(frame_paths: list[str], orig_fps: float) -> list[float]:
    try:
        nums = [int(re.findall(r"\d+", Path(p).stem)[-1]) for p in frame_paths]
        min_n = min(nums)
        return [(n - min_n) / orig_fps for n in nums]
    except Exception:
        return [i * 2.0 for i in range(len(frame_paths))]

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
    frame_timestamps: bool = False,
    orig_fps: float = 30.0,
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
        if frame_timestamps and item["images"]:
            timestamps = _compute_eval_timestamps(item["images"], orig_fps)
            parts = []
            for i, p in enumerate(item["images"]):
                parts.append(types.Part.from_text(text=f"[Frame at {timestamps[i]:.1f}s]"))
                parts.append(load_image_part(p))
        else:
            timestamps = None
            parts = [load_image_part(p) for p in item["images"]]
        parts.append(types.Part.from_text(text=item["prompt"]))
        contents.append(types.Content(role="user", parts=parts))
        turn = {"role": "user", "text": item["prompt"], "images": item["images"]}
        if timestamps is not None:
            turn["timestamps"] = timestamps
        turns.append(turn)

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
            feedback = (
                f"Correct. In 1-2 sentences, what key visual evidence from the frames "
                f"confirmed that {gt} is the right answer?"
            )
        else:
            feedback = (
                f"You answered {model_answer}, but the correct answer is {gt}. "
                f"(1) What specific visual evidence in the frames supports {gt}? "
                f"(2) Why does that evidence rule out {model_answer}? "
                f"Answer in 2-3 sentences."
            )
        contents.append(types.Content(role="user", parts=[
                        types.Part.from_text(text=feedback)]))
        turns.append({"role": "user", "text": feedback, "images": []})

        status = "✓" if correct else f"✗ pred={model_answer} gt={gt}"
        print(f"    {status}  {item['prompt'][:70]}")

        # 正解・不正解ともに根拠説明を取得
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
    parser.add_argument("--project", default="",
                        help="Vertex AI 使用時のGCPプロジェクトID (GOOGLE_CLOUD_PROJECT 環境変数でも可)")
    parser.add_argument("--location", default="global",
                        help="Vertex AI ロケーション (default: global)")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT),
                        help=f"出力 JSON パス (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--frame-timestamps", action="store_true",
                        help="temporal問題のフレームに '[Frame at X.Xs]' を追加してJSON保存")
    args = parser.parse_args()

    # Client setup
    if args.use_vertex:
        project = args.project or os.environ.get("GOOGLE_CLOUD_PROJECT", "")
        if not project:
            raise ValueError("--project または GOOGLE_CLOUD_PROJECT 環境変数が必要です")
        client = genai.Client(vertexai=True, project=project, location=args.location)
        print(f"Using Vertex AI (ADC, project={project}, location={args.location})")
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
        use_ts = args.frame_timestamps and "temporal" in qt.lower()
        turns, n_correct, n_total = build_warmup_for_group(
            client, group_items, args.model, config, args.rate_limit_sleep,
            frame_timestamps=use_ts, orig_fps=DOMAIN_ORIG_FPS.get(domain, 30.0))
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
