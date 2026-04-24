import argparse
import json
import re
import torch
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
from qwen_vl_utils import process_vision_info

BASE = Path(__file__).parent
TEST_JSON = BASE / "EgoCross_test/egocross_testbed/egocross_testbed_imgs.json"
SUPPORT_JSON = BASE / "data/egocross/train.json"
SUBMISSION_TEMPLATE = BASE / "../EgoCross_SFT_qwen3vl4b/submission_template.json"
IMAGE_BASE = BASE / "EgoCross_test"
MODEL_BASE = BASE / "models"
OUTPUT_DIR = BASE / "outputs"

# test set: dataset name → model name
DATASET_MODEL = {
    "CholecTrack20":   "surgery",
    "EgoSurgery":      "surgery",
    "ENIGMA":          "industry",
    "ExtrameSportFPV": "xsports",
    "EgoPet":          "animal",
}


def load_test_items() -> list[dict]:
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
        items.append({
            "id": d["id"],
            "images": [str(IMAGE_BASE / p.lstrip("/")) for p in d["video_path"]],
            "prompt": prompt,
            "domain": model_name,
            "ground_truth": None,
        })
    return items


def load_eval_items() -> list[dict]:
    with open(SUPPORT_JSON) as f:
        raw = json.load(f)
    items = []
    for i, d in enumerate(raw):
        user_text = re.sub(r"<image>", "", d["messages"][0]["content"]).strip()
        prompt = user_text + "\nAnswer with only a single letter: A, B, C, or D."
        items.append({
            "id": i,
            "images": d["images"],
            "prompt": prompt,
            "domain": d["domain"],
            "ground_truth": d["messages"][-1]["content"].strip().upper(),
        })
    return items


def run_domain(model_name: str, items: list, log_lines: list) -> dict:
    model_path = str(MODEL_BASE / model_name)
    msg = f"\n=== Loading model: {model_name} ({len(items)} questions) ==="
    print(msg)
    log_lines.append(msg)

    processor = Qwen3VLProcessor.from_pretrained(model_path)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    answers = {}
    for i, item in enumerate(items):
        content = [{"type": "image", "image": p} for p in item["images"]]
        content.append({"type": "text", "text": item["prompt"]})
        messages = [{"role": "user", "content": content}]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        image_inputs, _ = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs, max_new_tokens=4, do_sample=False)

        generated = output_ids[0][inputs.input_ids.shape[1]:]
        raw = processor.decode(generated, skip_special_tokens=True).strip()
        answer = raw[0].upper() if raw else "A"

        gt = item["ground_truth"]
        correct = f" ✓" if gt and answer == gt else (
            f" ✗(gt={gt})" if gt else "")
        answers[item["id"]] = answer

        if (i + 1) % 10 == 0 or i == 0:
            line = f"  [{i+1}/{len(items)}] id={item['id']} raw='{raw}' → {answer}{correct}"
            print(line)
            log_lines.append(line)

    del model
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
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_lines = [f"Run started: {timestamp}  mode={args.mode}"]

    if args.mode == "test":
        items = load_test_items()
        with open(SUBMISSION_TEMPLATE) as f:
            submission = json.load(f)
        id_to_entry = {e["id"]: e for e in submission}
    else:
        items = load_eval_items()

    by_model = defaultdict(list)
    for item in items:
        by_model[item["domain"]].append(item)

    all_answers = {}
    for model_name, domain_items in by_model.items():
        domain_answers = run_domain(model_name, domain_items, log_lines)
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
