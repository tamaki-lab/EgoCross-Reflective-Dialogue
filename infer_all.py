import argparse
import gc
import json
import re
import time
import torch
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info

BASE = Path(__file__).parent
TEST_JSON = BASE / "EgoCross_test/egocross_testbed/egocross_testbed_imgs.json"
SUPPORT_JSON = BASE / "data/egocross/train.json"
SUBMISSION_TEMPLATE = BASE / "../EgoCross_SFT_qwen3vl4b/submission_template.json"
IMAGE_BASE = BASE / "EgoCross_test"
MODEL_BASE = BASE / "models"
OUTPUT_DIR = BASE / "outputs"

_DOMAIN_FORMAT = " You must always commit to exactly one of the given options (A, B, C, or D) and end your response with a single line in the format: 'Final Answer: X' (X is one of A, B, C, or D). Do not refuse, do not say 'none of the above', and do not output multiple letters."

DOMAIN_SYSTEM = {
    "animal":   "You are an expert analyzing egocentric video frames featuring animals. Carefully observe the animal species and behaviors shown." + _DOMAIN_FORMAT,
    "industry": "You are an expert analyzing egocentric video frames from industrial or factory settings. Carefully observe the tools, machinery, and work activities shown." + _DOMAIN_FORMAT,
    "xsports":  "You are an expert analyzing egocentric video frames from extreme sports. Carefully observe the sport type, actions, and environment shown." + _DOMAIN_FORMAT,
    "surgery":  "You are an expert analyzing egocentric video frames from surgical procedures. Carefully observe the instruments, tissues, and surgical actions shown." + _DOMAIN_FORMAT,
}

# test set: dataset name → model name
DATASET_MODEL = {
    "CholecTrack20":   "surgery",
    "EgoSurgery":      "surgery",
    "ENIGMA":          "industry",
    "ExtrameSportFPV": "xsports",
    "EgoPet":          "animal",
}


def load_test_items(fewshot: bool = False) -> list[dict]:
    # train.json全件をtrain bankとして構築
    train_bank: dict[str, list] = {}
    if fewshot:
        with open(SUPPORT_JSON) as f:
            train_raw = json.load(f)
        for d in train_raw:
            user_text = re.sub(
                r"<image>", "", d["messages"][0]["content"]).strip()
            q_key = user_text.split("\nA.")[0].strip()
            train_bank.setdefault(q_key, []).append({
                "prompt": user_text,
                "answer": d["messages"][-1]["content"].strip().upper(),
            })

    with open(TEST_JSON) as f:
        raw = json.load(f)
    items = []
    for d in raw:
        model_name = DATASET_MODEL.get(d["dataset"])
        if model_name is None:
            print(f"WARNING: unknown dataset '{d['dataset']}'")
            continue
        options = "\n".join(d["options"])
        prompt = (
            f"{d['question_text']}\n{options}\n"
            "Answer with only a single letter: A, B, C, or D."
        )
        q_key = d["question_text"].strip()
        items.append({
            "id": d["id"],
            "images": [str(IMAGE_BASE / p.lstrip("/")) for p in d["video_path"]],
            "prompt": prompt,
            "domain": model_name,
            "ground_truth": None,
            "fps": d.get("original_video_fps", 1.0),
            "fewshot_examples": train_bank.get(q_key, []),
        })
    return items


def load_eval_items(prompt_style: str = "default", fewshot: bool = False) -> list[dict]:
    with open(SUPPORT_JSON) as f:
        raw = json.load(f)

    if fewshot:
        # 問題文でグループ化し、前半をtrain bank・後半をevalに分割
        by_question = defaultdict(list)
        for i, d in enumerate(raw):
            user_text = re.sub(
                r"<image>", "", d["messages"][0]["content"]).strip()
            q_key = user_text.split("\nA.")[0].strip()
            by_question[q_key].append((i, d, user_text))

        train_bank: dict[str, list] = {}
        eval_entries = []
        for q_key, group in by_question.items():
            mid = max(1, len(group) // 2)
            for (i, d, user_text) in group[:mid]:
                train_bank.setdefault(q_key, []).append({
                    "prompt": user_text,
                    "answer": d["messages"][-1]["content"].strip().upper(),
                })
            for (i, d, user_text) in group[mid:]:
                eval_entries.append(
                    (i, d, user_text, train_bank.get(q_key, [])))

        items = []
        for (i, d, user_text, examples) in eval_entries:
            if prompt_style == "default":
                prompt = user_text + "\nAnswer with only a single letter: A, B, C, or D."
            elif prompt_style == "domain":
                prompt = user_text + "\nYou MUST pick exactly one option. End your response with a single line: 'Final Answer: X' where X is one of A, B, C, or D."
            else:
                prompt = user_text
            items.append({
                "id": i,
                "images": d["images"],
                "prompt": prompt,
                "domain": d["domain"],
                "ground_truth": d["messages"][-1]["content"].strip().upper(),
                "fps": 1.0,
                "fewshot_examples": examples,
            })
        return items

    items = []
    for i, d in enumerate(raw):
        user_text = re.sub(r"<image>", "", d["messages"][0]["content"]).strip()
        if prompt_style == "default":
            prompt = user_text + "\nAnswer with only a single letter: A, B, C, or D."
        elif prompt_style == "domain":
            prompt = user_text + "\nYou MUST pick exactly one option. End your response with a single line: 'Final Answer: X' where X is one of A, B, C, or D."
        else:
            prompt = user_text
        items.append({
            "id": i,
            "images": d["images"],
            "prompt": prompt,
            "domain": d["domain"],
            "ground_truth": d["messages"][-1]["content"].strip().upper(),
            "fps": 1.0,
            "fewshot_examples": [],
        })
    return items


def run_domain(model_name: str, items: list, log_lines: list, max_pixels: int = 360000, input_mode: str = "image", thinking: bool = False, baseline: bool = False, prompt_style: str = "default", single_model: str = "", model_id: str = "") -> dict:
    if model_id:
        model_path = model_id
    elif baseline:
        model_path = "Qwen/Qwen3-VL-4B-Instruct"
    elif single_model:
        model_path = str(MODEL_BASE / single_model)
    else:
        model_path = str(MODEL_BASE / model_name)
    msg = f"\n=== Loading model: {model_path} ({len(items)} questions) ==="
    print(msg)
    log_lines.append(msg)

    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        # force single GPU; multi-GPU split causes NaN on Turing
        device_map={"": 0},
    )
    model.eval()
    print(
        f"Model: {model.__class__.__name__}, device_map: {getattr(model, 'hf_device_map', None)}")

    answers = {}
    pbar = tqdm(items, desc=model_name, unit="q")
    for item in pbar:
        t0 = time.time()
        image_paths = item["images"]

        raw = None
        for attempt in range(3):
            try:
                # few-shot examples をテキストのみでプロンプト先頭に付加
                fewshot_text = ""
                for ex in item.get("fewshot_examples", []):
                    fewshot_text += f"{ex['prompt']}\nAnswer: {ex['answer']}\n\n"

                if input_mode == "video":
                    content = [{
                        "type": "video", "video": image_paths,
                        "fps": item.get("fps", 1.0),
                        "min_pixels": 50176, "max_pixels": max_pixels,
                    }]
                else:
                    content = [{
                        "type": "image", "image": p,
                        "min_pixels": 50176, "max_pixels": max_pixels,
                    } for p in image_paths]
                if fewshot_text:
                    content.insert(0, {"type": "text", "text": fewshot_text})
                content.append({"type": "text", "text": item["prompt"]})
                if prompt_style == "domain":
                    system_text = DOMAIN_SYSTEM.get(item["domain"], "")
                    messages = [{"role": "system", "content": system_text}, {
                        "role": "user", "content": content}]
                else:
                    messages = [{"role": "user", "content": content}]

                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=thinking,
                )
                image_inputs, video_inputs, video_kwargs = process_vision_info(
                    messages, return_video_kwargs=True
                )
                # process_vision_info returns fps as a list; processor expects a scalar
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
                    max_new = 2048 if thinking else 32
                    output_ids = model.generate(
                        **inputs, max_new_tokens=max_new, do_sample=False, use_cache=True)
                    generated = output_ids[0][inputs.input_ids.shape[1]:]
                    raw = processor.decode(
                        generated, skip_special_tokens=True).strip()
                    del output_ids, generated
                del inputs, image_inputs, video_inputs
                break
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                image_paths = image_paths[::2] or image_paths[:1]  # フレームを半分に
                pbar.write(
                    f"  OOM id={item['id']}, retry with {len(image_paths)} frames")

        if raw is None:
            raw = ""
        answer_text = re.sub(
            r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        answer = None
        for pat in (
            r"Final\s*Answer\s*[:：]\s*\**\s*\(?\s*([A-D])",
            r"answer\s+is\s*[:：]?\s*\**\s*\(?\s*([A-D])",
            r"correct\s+(?:option|choice)\s+is\s*[:：]?\s*\**\s*\(?\s*([A-D])",
        ):
            m = re.search(pat, answer_text, re.IGNORECASE)
            if m:
                answer = m.group(1).upper()
                break
        if answer is None:
            # No Final Answer pattern — follow-up text-only query to get a committed letter
            try:
                followup_messages = [
                    {"role": "user", "content": [
                        {"type": "text", "text": item["prompt"]}]},
                    {"role": "assistant", "content": answer_text[:500]},
                    {"role": "user", "content": [
                        {"type": "text", "text": "Answer with only a single letter: A, B, C, or D."}]},
                ]
                fu_text = processor.apply_chat_template(
                    followup_messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False,
                )
                fu_inputs = processor(
                    text=[fu_text], padding=True, return_tensors="pt",
                ).to(model.device)
                with torch.no_grad():
                    fu_ids = model.generate(
                        **fu_inputs, max_new_tokens=10, do_sample=False)
                    fu_gen = fu_ids[0][fu_inputs.input_ids.shape[1]:]
                    fu_raw = processor.decode(
                        fu_gen, skip_special_tokens=True).strip()
                del fu_ids, fu_gen, fu_inputs
                pbar.write(f"  follow-up id={item['id']} raw='{fu_raw}'")
                log_lines.append(f"  follow-up id={item['id']} raw='{fu_raw}'")
                m = re.search(r"\b([A-D])\b", fu_raw, re.IGNORECASE)
                if m:
                    answer = m.group(1).upper()
            except Exception as e:
                pbar.write(f"  follow-up failed id={item['id']}: {e}")
        if answer is None:
            matches = re.findall(r"\b([A-D])\b", answer_text)
            answer = matches[-1] if matches else (
                answer_text[0].upper() if answer_text and answer_text[0].upper() in "ABCD" else "A")

        gt = item["ground_truth"]
        correct = " ✓" if gt and answer == gt else (
            f" ✗(gt={gt})" if gt else "")
        answers[item["id"]] = answer

        gc.collect()
        torch.cuda.empty_cache()

        alloc = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        elapsed = time.time() - t0
        probs_str = " probs=" + \
            str(item.get("_probs", {})) if "_probs" in item else ""
        line = f"id={item['id']} raw='{raw}' → {answer}{correct}{probs_str} ({elapsed:.1f}s) VRAM alloc={alloc:.1f}GB reserved={reserved:.1f}GB"
        pbar.write(line)
        log_lines.append(line)

    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    return answers


def print_accuracy(items: list, all_answers: dict, log_lines: list):
    by_domain = defaultdict(lambda: {"correct": 0, "total": 0})
    for item in items:
        pred = all_answers.get(item["id"], "A")
        gt = item["ground_truth"]
        domain = item["domain"]
        by_domain[domain]["total"] += 1
        if pred == gt:
            by_domain[domain]["correct"] += 1

    total_c, total_n = 0, 0
    lines = ["\n=== Accuracy ==="]
    for domain, stat in sorted(by_domain.items()):
        c, n = stat["correct"], stat["total"]
        lines.append(f"  {domain:10s}: {c}/{n} = {c/n*100:.1f}%")
        total_c += c
        total_n += n
    lines.append(
        f"  {'Overall':10s}: {total_c}/{total_n} = {total_c/total_n*100:.1f}%")
    for line in lines:
        print(line)
        log_lines.append(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["test", "eval"], default="test",
                        help="test: 提出用予測 / eval: サポートセットで正解率確認")
    parser.add_argument("--max-pixels", type=int, default=128000,
                        help="1フレームあたりの最大ピクセル数 (default: 128000)")
    parser.add_argument("--input-mode", choices=["image", "video"], default="image",
                        help="image: フレームを個別画像として入力 / video: 動画として入力")
    parser.add_argument("--thinking", action="store_true",
                        help="thinkingモードを有効化")
    parser.add_argument("--baseline", action="store_true",
                        help="fine-tunedモデルではなくベースモデル(Qwen3-VL-4B-Instruct)を使用")
    parser.add_argument("--prompt-style", choices=["default", "clean", "domain"], default="default",
                        help="default: 末尾に指示あり / clean: 学習時と同じ形式 / domain: ドメイン別システムプロンプト付き")
    parser.add_argument("--fewshot", action="store_true",
                        help="同じ問題文の学習例をテキストfew-shotとして使用（eval: 前半train/後半eval分割, test: train.json全件をbank）")
    parser.add_argument("--single-model", default="",
                        help="全ドメインで同一モデルを使用 (例: --single-model egocross)")
    parser.add_argument("--model-id", default="",
                        help="HuggingFace HubのモデルIDを直接指定 (例: --model-id Qwen/Qwen3.6-27B)。指定時は --baseline / --single-model より優先")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_lines = [
        f"Run started: {timestamp}  mode={args.mode}  max_pixels={args.max_pixels}  input_mode={args.input_mode}  thinking={args.thinking}  baseline={args.baseline}  prompt_style={args.prompt_style}  fewshot={args.fewshot}  single_model={args.single_model!r}  model_id={args.model_id!r}"]

    if args.mode == "test":
        items = load_test_items(fewshot=args.fewshot)
        with open(SUBMISSION_TEMPLATE) as f:
            submission = json.load(f)
        id_to_entry = {e["id"]: e for e in submission}
    else:
        items = load_eval_items(args.prompt_style, fewshot=args.fewshot)

    by_model = defaultdict(list)
    for item in items:
        by_model[item["domain"]].append(item)

    all_answers = {}
    for model_name, domain_items in by_model.items():
        domain_answers = run_domain(
            model_name, domain_items, log_lines, args.max_pixels, args.input_mode,
            thinking=args.thinking, baseline=args.baseline, prompt_style=args.prompt_style,
            single_model=args.single_model, model_id=args.model_id)
        all_answers.update(domain_answers)

    if args.mode == "test":
        for entry in submission:
            entry["answer"] = all_answers.get(entry["id"], "A")
        pred_path = OUTPUT_DIR / f"predictions_{timestamp}.json"
        with open(pred_path, "w") as f:
            json.dump(submission, f, indent=2)
        line = f"\nSaved {len(submission)} predictions → {pred_path}"
        print(line)
        log_lines.append(line)
    else:
        print_accuracy(items, all_answers, log_lines)

    log_path = OUTPUT_DIR / f"log_{args.mode}_{timestamp}.txt"
    with open(log_path, "w") as f:
        f.write("\n".join(log_lines))
    print(f"Log → {log_path}")


if __name__ == "__main__":
    main()
