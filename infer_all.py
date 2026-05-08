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

_DOMAIN_BASE = {
    "animal":   "You are an expert analyzing egocentric video frames featuring animals. Carefully observe the animal species and behaviors shown.",
    "industry": "You are an expert analyzing egocentric video frames from industrial or factory settings. Carefully observe the tools, machinery, and work activities shown.",
    "xsports":  "You are an expert analyzing egocentric video frames from extreme sports. Carefully observe the sport type, actions, and environment shown.",
    "surgery":  "You are an expert analyzing egocentric video frames from surgical procedures. Carefully observe the instruments, tissues, and surgical actions shown.",
}

DOMAIN_SYSTEM = {k: v + _DOMAIN_FORMAT for k, v in _DOMAIN_BASE.items()}


def build_warmup_system(domain: str, question_type: str) -> str:
    base = _DOMAIN_BASE.get(
        domain, f"You are an expert analyzing egocentric video frames from the '{domain}' domain.")
    return (
        f"{base}\n\n"
        f"The conversation begins with warm-up practice questions of type '{question_type}'. "
        "Each warm-up question is followed by correct/incorrect feedback and, if the answer was wrong, "
        "a reflection on the mistake. "
        "Learn from these reflections to improve your performance on similar questions.\n\n"
        "After the warm-up you will see 'Warm-up complete. Now answer the following question:' — "
        "that is the actual question you must answer."
        + _DOMAIN_FORMAT
    )

# test set: dataset name → model name
DATASET_MODEL = {
    "CholecTrack20":   "surgery",
    "EgoSurgery":      "surgery",
    "ENIGMA":          "industry",
    "ExtrameSportFPV": "xsports",
    "EgoPet":          "animal",
}


def _subsample_frames(images: list[str], max_frames: int) -> list[str]:
    if max_frames <= 0 or len(images) <= max_frames:
        return images
    step = (len(images) - 1) / (max_frames - 1) if max_frames > 1 else 0
    return [images[round(i * step)] for i in range(max_frames)]


CLASSIFY_JSON = BASE / "outputs/support_question_types.json"


def _load_support_question_types() -> dict[int, str]:
    if not CLASSIFY_JSON.exists():
        return {}
    with open(CLASSIFY_JSON) as f:
        data = json.load(f)
    return {entry["index"]: entry["predicted_type"] for entry in data}


def load_warmup_contents(path: str, max_frames: int = 0, max_pixels: int = 360000) -> dict[str, list[dict]]:
    """warmup_conversations.json を読み込み、HuggingFace messages 形式を返す。
    key は 'domain::question_type' 形式。"""
    with open(path) as f:
        data = json.load(f)
    result = {}
    for key, group in data.items():
        domain, qt = key.split("::", 1)
        preamble = (
            f"The following are warm-up practice questions of type '{qt}' "
            f"in the '{domain}' domain. "
            "After each question you will see whether the answer was correct and, "
            "if incorrect, a reflection on the mistake. "
            "Study these carefully to improve your performance on similar questions."
        )
        messages = [
            {"role": "user", "content": [{"type": "text", "text": preamble}]},
            {"role": "assistant", "content": "Understood. I will study these practice questions and reflections carefully."},
        ]
        for turn in group["turns"]:
            if turn["role"] == "user":
                images = turn.get("images", [])
                if max_frames > 0 and len(images) > max_frames:
                    step = (len(images) - 1) / (max_frames - 1) if max_frames > 1 else 0
                    images = [images[round(i * step)] for i in range(max_frames)]
                content = [
                    {"type": "image", "image": p, "min_pixels": 50176, "max_pixels": max_pixels}
                    for p in images
                ]
                if turn.get("text"):
                    content.append({"type": "text", "text": turn["text"]})
                messages.append({"role": "user", "content": content})
            else:
                messages.append({"role": "assistant", "content": turn["text"]})
        result[key] = messages
    frame_info = f", max_frames={max_frames}" if max_frames > 0 else ""
    print(f"Loaded warmup for {len(result)} groups from {path}{frame_info}")
    return result


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
                "images": d["images"],
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
            "question_type": d.get("question_type", ""),
            "ground_truth": None,
            "fps": d.get("original_video_fps", 1.0),
            "fewshot_examples": train_bank.get(q_key, []),
        })
    return items


def load_eval_items(prompt_style: str = "default", fewshot: bool = False) -> list[dict]:
    with open(SUPPORT_JSON) as f:
        raw = json.load(f)

    qt_map = _load_support_question_types()

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
                    "images": d["images"],
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
                "question_type": qt_map.get(i, ""),
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
            "question_type": qt_map.get(i, ""),
            "ground_truth": d["messages"][-1]["content"].strip().upper(),
            "fps": 1.0,
            "fewshot_examples": [],
        })
    return items


def run_domain(model_name: str, items: list, log_lines: list, max_pixels: int = 360000, input_mode: str = "image", thinking: bool = False, baseline: bool = False, prompt_style: str = "default", single_model: str = "", model_id: str = "", warmup_contents: dict | None = None, visual_fewshot: bool = False, visual_fewshot_max_frames: int = 0) -> dict:
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
        attn_implementation="sdpa",
        # force single GPU; multi-GPU split causes NaN on Turing
        device_map={"": 0},
    )
    model.eval()
    print(
        f"Model: {model.__class__.__name__}, device_map: {getattr(model, 'hf_device_map', None)}")

    torch.cuda.reset_peak_memory_stats()
    domain_t0 = time.time()
    answers = {}
    pbar = tqdm(items, desc=model_name, unit="q")
    for item in pbar:
        t0 = time.time()
        image_paths = item["images"]

        raw = None
        for attempt in range(3):
            try:
                # few-shot は warmup/visual_fewshot 未使用時のみ
                fewshot_text = ""
                if not warmup_contents and not visual_fewshot:
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

                warmup_key = f"{item['domain']}::{item.get('question_type', '')}" if warmup_contents else ""
                prefix = warmup_contents.get(warmup_key, []) if warmup_key else []
                if prefix:
                    system_text = build_warmup_system(item["domain"], item.get("question_type", ""))
                    separator = [{"type": "text", "text": "Warm-up complete. Now answer the following question:"}]
                    messages = [{"role": "system", "content": system_text}] + prefix + [
                        {"role": "user", "content": separator + content}
                    ]
                elif visual_fewshot and item.get("fewshot_examples"):
                    vf_messages: list[dict] = []
                    if prompt_style == "domain":
                        system_text = DOMAIN_SYSTEM.get(item["domain"], "")
                        vf_messages = [{"role": "system", "content": system_text}]
                    for ex in item["fewshot_examples"]:
                        ex_imgs = _subsample_frames(ex.get("images", []), visual_fewshot_max_frames)
                        ex_content = [
                            {"type": "image", "image": p, "min_pixels": 50176, "max_pixels": max_pixels}
                            for p in ex_imgs
                        ]
                        ex_content.append({"type": "text", "text": ex["prompt"]})
                        vf_messages.append({"role": "user", "content": ex_content})
                        vf_messages.append({"role": "assistant", "content": f"Final Answer: {ex['answer']}"})
                    vf_messages.append({"role": "user", "content": content})
                    messages = vf_messages
                elif prompt_style == "domain":
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

    peak_vram = torch.cuda.max_memory_allocated() / 1024**3
    domain_elapsed = time.time() - domain_t0
    summary = (
        f"  [{model_name}] peak VRAM={peak_vram:.2f}GB"
        f"  total time={domain_elapsed:.1f}s ({domain_elapsed/60:.1f}min)"
        f"  questions={len(items)}"
    )
    print(summary)
    log_lines.append(summary)

    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    return answers, peak_vram, domain_elapsed


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
    parser.add_argument("--warmup-file", default="",
                        help="warmup_conversations.json のパス (指定時: warm-up 会話を前置して推論)")
    parser.add_argument("--warmup-max-frames", type=int, default=0,
                        help="warmup 各ターンの画像フレーム上限 (0=制限なし, default: 0)")
    parser.add_argument("--visual-fewshot", action="store_true",
                        help="support setの画像+問題+回答をそのままvisual few-shotとして渡す (reflection/thinkingなし)")
    parser.add_argument("--visual-fewshot-max-frames", type=int, default=0,
                        help="visual few-shot 各例の画像フレーム上限 (0=制限なし, default: 0)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_lines = [
        f"Run started: {timestamp}  mode={args.mode}  max_pixels={args.max_pixels}  input_mode={args.input_mode}  thinking={args.thinking}  baseline={args.baseline}  prompt_style={args.prompt_style}  fewshot={args.fewshot}  single_model={args.single_model!r}  model_id={args.model_id!r}  warmup_file={args.warmup_file!r}"]

    warmup_contents = None
    if args.warmup_file:
        warmup_contents = load_warmup_contents(
            args.warmup_file, max_frames=args.warmup_max_frames, max_pixels=args.max_pixels)

    need_fewshot_bank = args.fewshot or args.visual_fewshot
    if args.mode == "test":
        items = load_test_items(fewshot=need_fewshot_bank)
        with open(SUBMISSION_TEMPLATE) as f:
            submission = json.load(f)
        id_to_entry = {e["id"]: e for e in submission}
    else:
        items = load_eval_items(args.prompt_style, fewshot=need_fewshot_bank)

    by_model = defaultdict(list)
    for item in items:
        by_model[item["domain"]].append(item)

    run_t0 = time.time()
    all_answers = {}
    grand_peak_vram = 0.0
    grand_elapsed = 0.0
    for model_name, domain_items in by_model.items():
        domain_answers, peak_vram, domain_elapsed = run_domain(
            model_name, domain_items, log_lines, args.max_pixels, args.input_mode,
            thinking=args.thinking, baseline=args.baseline, prompt_style=args.prompt_style,
            single_model=args.single_model, model_id=args.model_id,
            warmup_contents=warmup_contents,
            visual_fewshot=args.visual_fewshot,
            visual_fewshot_max_frames=args.visual_fewshot_max_frames)
        all_answers.update(domain_answers)
        grand_peak_vram = max(grand_peak_vram, peak_vram)
        grand_elapsed += domain_elapsed

    total_wall = time.time() - run_t0
    run_summary = (
        f"\n=== Run summary ==="
        f"\n  Peak VRAM (max across domains): {grand_peak_vram:.2f} GB"
        f"\n  Total inference time: {grand_elapsed:.1f}s ({grand_elapsed/60:.1f}min)"
        f"\n  Wall time: {total_wall:.1f}s ({total_wall/60:.1f}min)"
    )
    print(run_summary)
    log_lines.append(run_summary)

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
