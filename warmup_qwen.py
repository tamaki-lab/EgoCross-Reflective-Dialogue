"""
Qwen ローカルモデルで warm-up 会話を構築する。
outputs/support_question_types.json (classify_support.py の出力) を読み込み、
各 (domain, question_type) グループで順番に問題を解かせ、
正解・不正解のフィードバック＋反省を含む会話を構築して
outputs/warmup_conversations_qwen.json に保存する。
"""
import argparse
import gc
import json
import re
import torch
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info

BASE = Path(__file__).parent
TRAIN_JSON = BASE / "data/egocross/train.json"
CLASSIFY_JSON = BASE / "outputs/support_question_types.json"
DEFAULT_OUTPUT = BASE / "outputs/warmup_conversations_qwen.json"
MODEL_BASE = BASE / "models"


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


def call_model(model, processor, messages: list, max_new_tokens: int = 512, thinking: bool = False) -> str:
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=thinking,
    )
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages, return_video_kwargs=True)
    if isinstance(video_kwargs.get("fps"), list):
        video_kwargs["fps"] = video_kwargs["fps"][0] if video_kwargs["fps"] else None
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)
        generated = output_ids[0][inputs.input_ids.shape[1]:]
        raw = processor.decode(generated, skip_special_tokens=True).strip()
        del output_ids, generated
    del inputs, image_inputs, video_inputs
    gc.collect()
    torch.cuda.empty_cache()
    return raw


def build_warmup_for_group(
    model,
    processor,
    group_items: list[dict],
    max_pixels: int,
    thinking: bool,
    explain_correct: bool = False,
) -> tuple[list[dict], int, int]:
    """
    1グループ分の warm-up 会話を構築する。

    Returns:
        turns: JSON保存用のターンリスト (role, text, images フィールド)
        n_correct: 正解数
        n_total: 問題数
    """
    messages: list[dict] = []
    turns: list[dict] = []
    n_correct = 0

    for item in group_items:
        # User turn: 画像 + 問題文
        content = [
            {"type": "image", "image": p, "min_pixels": 50176, "max_pixels": max_pixels}
            for p in item["images"]
        ]
        content.append({"type": "text", "text": item["prompt"]})
        messages.append({"role": "user", "content": content})
        turns.append({"role": "user", "text": item["prompt"], "images": item["images"]})

        # Model の回答
        raw = call_model(model, processor, messages,
                         max_new_tokens=2048 if thinking else 256, thinking=thinking)
        answer_text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        if not answer_text:
            answer_text = "Final Answer: A"
        model_answer = extract_answer(answer_text)
        messages.append({"role": "assistant", "content": answer_text})
        turns.append({"role": "model", "text": answer_text})

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
        messages.append({"role": "user", "content": [{"type": "text", "text": feedback}]})
        turns.append({"role": "user", "text": feedback, "images": []})

        status = "✓" if correct else f"✗ pred={model_answer} gt={gt}"
        print(f"    {status}  {item['prompt'][:70]}")

        # 不正解は反省、正解かつ explain_correct は根拠説明を取得
        if not correct or explain_correct:
            try:
                response = call_model(model, processor, messages,
                                      max_new_tokens=512, thinking=False)
                response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                response = "I will be more careful next time." if not correct else ""
            if response:
                messages.append({"role": "assistant", "content": response})
                turns.append({"role": "model", "text": response})

    return turns, n_correct, len(group_items)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="",
                        help="models/ 以下のモデルディレクトリ名")
    parser.add_argument("--baseline", action="store_true",
                        help="ベースモデル (Qwen/Qwen3-VL-4B-Instruct) を使用")
    parser.add_argument("--model-id", default="",
                        help="HuggingFace Hub のモデルID (指定時は --model / --baseline より優先)")
    parser.add_argument("--max-pixels", type=int, default=128000,
                        help="1フレームあたりの最大ピクセル数 (default: 128000)")
    parser.add_argument("--thinking", action="store_true",
                        help="thinking モードを有効化")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT),
                        help=f"出力 JSON パス (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--explain-correct", action="store_true",
                        help="正解時にも根拠説明をモデルに生成させる (default: off)")
    args = parser.parse_args()

    if args.model_id:
        model_path = args.model_id
    elif args.baseline:
        model_path = "Qwen/Qwen3-VL-4B-Instruct"
    elif args.model:
        model_path = str(MODEL_BASE / args.model)
    else:
        raise ValueError("--model / --baseline / --model-id のいずれかを指定してください")

    print(f"Loading model: {model_path}")
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map={"": 0},
    )
    model.eval()

    if not CLASSIFY_JSON.exists():
        raise FileNotFoundError(
            f"{CLASSIFY_JSON} が見つかりません。先に classify_support.py を実行してください。")
    with open(CLASSIFY_JSON) as f:
        classify_results = json.load(f)
    with open(TRAIN_JSON) as f:
        train_data = json.load(f)

    assert len(classify_results) == len(train_data), "classify と train のサイズが一致しません"

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

    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for item in merged:
        groups[(item["domain"], item["question_type"])].append(item)

    print(f"\n{len(groups)} グループ:")
    for (domain, qt), items in sorted(groups.items()):
        print(f"  [{domain:10s}] {qt:45s} {len(items)}問")

    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True)

    warmup_data = {}
    grand_correct = grand_total = 0

    for (domain, qt), group_items in sorted(groups.items()):
        key = f"{domain}::{qt}"
        print(f"\n=== {key} ({len(group_items)}問) ===")
        turns, n_correct, n_total = build_warmup_for_group(
            model, processor, group_items, args.max_pixels, args.thinking,
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
    print(f"  Warm-up accuracy: {grand_correct}/{grand_total} = {grand_correct/grand_total*100:.1f}%")
    print(f"  Saved → {output_path}")


if __name__ == "__main__":
    main()
