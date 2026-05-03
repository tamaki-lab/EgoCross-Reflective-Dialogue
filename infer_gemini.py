import argparse
import json
import os
import re
import subprocess
import tempfile
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from google import genai
from google.genai import types

BASE = Path(__file__).parent
TEST_JSON = BASE / "EgoCross_test/egocross_testbed/egocross_testbed_imgs.json"
SUPPORT_JSON = BASE / "data/egocross/train.json"
SUBMISSION_TEMPLATE = BASE / "../EgoCross_SFT_qwen3vl4b/submission_template.json"
IMAGE_BASE = BASE / "EgoCross_test"
OUTPUT_DIR = BASE / "outputs"

_DOMAIN_FORMAT = (
    " Output only: 'Final Answer: X' where X is A, B, C, or D. No explanation."
)

DOMAIN_SYSTEM = {
    "animal":   "You are an expert analyzing egocentric video frames featuring animals. Carefully observe the animal species and behaviors shown." + _DOMAIN_FORMAT,
    "industry": "You are an expert analyzing egocentric video frames from industrial or factory settings. Carefully observe the tools, machinery, and work activities shown." + _DOMAIN_FORMAT,
    "xsports":  "You are an expert analyzing egocentric video frames from extreme sports. Carefully observe the sport type, actions, and environment shown." + _DOMAIN_FORMAT,
    "surgery":  "You are an expert analyzing egocentric video frames from surgical procedures. Carefully observe the instruments, tissues, and surgical actions shown." + _DOMAIN_FORMAT,
}

DATASET_MODEL = {
    "CholecTrack20":   "surgery",
    "EgoSurgery":      "surgery",
    "ENIGMA":          "industry",
    "ExtrameSportFPV": "xsports",
    "EgoPet":          "animal",
}

# Gemini 3.x系はthinking必須 (budget=0はAPI拒否)


def requires_thinking(model: str) -> bool:
    return model.startswith("gemini-3")


MIME_MAP = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".png": "image/png", ".gif": "image/gif", ".webp": "image/webp",
}

DEFAULT_MODEL = "gemini-3.1-flash-image-preview"

# Pricing (USD per 1M tokens) — update if pricing changes
# https://cloud.google.com/vertex-ai/generative-ai/pricing
# output price includes reasoning tokens (no separate thinking price)
PRICING = {
    "gemini-3.1-flash-image-preview": {"input": 0.50, "output": 3.00,  "thinking": 0.0},
    "gemini-3.1-pro-preview":         {"input": 2.00, "output": 12.00, "thinking": 0.0},
    "gemini-3-pro-preview":           {"input": 2.00, "output": 12.00, "thinking": 0.0},
    "gemini-2.5-flash":               {"input": 0.075, "output": 0.30, "thinking": 3.50},
    "gemini-2.5-pro":                 {"input": 1.25,  "output": 10.00, "thinking": 3.50},
    "gemini-2.5-flash-lite":          {"input": 0.01,  "output": 0.04,  "thinking": 3.50},
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int, thinking_tokens: int) -> float:
    p = PRICING.get(model, PRICING["gemini-2.5-flash"])
    return (
        input_tokens / 1_000_000 * p["input"] +
        output_tokens / 1_000_000 * p["output"] +
        thinking_tokens / 1_000_000 * p["thinking"]
    )


def load_image_part(path: str) -> types.Part:
    p = Path(path)
    mime = MIME_MAP.get(p.suffix.lower(), "image/jpeg")
    return types.Part.from_bytes(data=p.read_bytes(), mime_type=mime)


def frames_to_video(frame_paths: list[str], fps: float = 1.0) -> str:
    """フレーム画像リストからmp4を作成し、一時ファイルパスを返す。"""
    tmp_list = tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False)
    duration = 1.0 / max(fps, 0.1)
    for p in frame_paths:
        tmp_list.write(f"file '{p}'\nduration {duration:.4f}\n")
    if frame_paths:
        # concat demuxer requires last entry without duration
        tmp_list.write(f"file '{frame_paths[-1]}'\n")
    tmp_list.flush()
    tmp_list.close()

    out_path = tmp_list.name.replace(".txt", ".mp4")
    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", tmp_list.name,
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            out_path,
        ],
        check=True, capture_output=True,
    )
    os.unlink(tmp_list.name)
    return out_path


def upload_video(client, video_path: str) -> str:
    """Gemini Files APIに動画をアップロードし、file nameを返す。"""
    with open(video_path, "rb") as f:
        uploaded = client.files.upload(
            file=f,
            config=types.UploadFileConfig(mime_type="video/mp4"),
        )
    # PROCESSING状態が終わるまで待機
    while True:
        info = client.files.get(name=uploaded.name)
        if info.state.name != "PROCESSING":
            break
        time.sleep(1)
    return uploaded.name, info.uri


def load_warmup_contents(path: str, max_frames: int = 0) -> dict[str, list[types.Content]]:
    """warmup_conversations.json を読み込み、画像バイトを復元して Content リストを返す。
    key は 'domain::question_type' 形式。
    max_frames > 0 のとき、各ユーザーターンの画像を先頭 max_frames 枚に制限する。"""
    with open(path) as f:
        data = json.load(f)
    result = {}
    for key, group in data.items():
        contents = []
        for turn in group["turns"]:
            if turn["role"] == "user":
                images = turn.get("images", [])
                if max_frames > 0:
                    images = images[:max_frames]
                parts = [load_image_part(p) for p in images]
                if turn.get("text"):
                    parts.append(types.Part.from_text(text=turn["text"]))
                contents.append(types.Content(role="user", parts=parts))
            else:
                contents.append(types.Content(
                    role="model",
                    parts=[types.Part.from_text(text=turn["text"])],
                ))
        result[key] = contents
    frame_info = f", max_frames={max_frames}" if max_frames > 0 else ""
    print(f"Loaded warmup for {len(result)} groups from {path}{frame_info}")
    return result


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


def load_test_items(fewshot: bool = False) -> list[dict]:
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
            "Output only: 'Final Answer: X' where X is A, B, C, or D. No explanation."
        )
        q_key = d["question_text"].strip()
        items.append({
            "id": d["id"],
            "images": [str(IMAGE_BASE / p.lstrip("/")) for p in d["video_path"]],
            "prompt": prompt,
            "domain": model_name,
            "question_type": d.get("question_type", ""),
            "ground_truth": None,
            "fewshot_examples": train_bank.get(q_key, []),
        })
    return items


CLASSIFY_JSON = BASE / "outputs/support_question_types.json"


def _load_support_question_types() -> dict[int, str]:
    """support_question_types.json が存在すれば index→question_type を返す。"""
    if not CLASSIFY_JSON.exists():
        return {}
    with open(CLASSIFY_JSON) as f:
        data = json.load(f)
    return {entry["index"]: entry["predicted_type"] for entry in data}


def load_eval_items(fewshot: bool = False) -> list[dict]:
    with open(SUPPORT_JSON) as f:
        raw = json.load(f)

    qt_map = _load_support_question_types()

    if fewshot:
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
            prompt = user_text + "\nYou MUST pick exactly one option. End your response with a single line: 'Final Answer: X' where X is one of A, B, C, or D."
            items.append({
                "id": i,
                "images": d["images"],
                "prompt": prompt,
                "domain": d["domain"],
                "question_type": qt_map.get(i, ""),
                "ground_truth": d["messages"][-1]["content"].strip().upper(),
                "fewshot_examples": examples,
            })
        return items

    items = []
    for i, d in enumerate(raw):
        user_text = re.sub(r"<image>", "", d["messages"][0]["content"]).strip()
        prompt = user_text + "\nYou MUST pick exactly one option. End your response with a single line: 'Final Answer: X' where X is one of A, B, C, or D."
        items.append({
            "id": i,
            "images": d["images"],
            "prompt": prompt,
            "domain": d["domain"],
            "question_type": qt_map.get(i, ""),
            "ground_truth": d["messages"][-1]["content"].strip().upper(),
            "fewshot_examples": [],
        })
    return items


def run_domain(
    client,
    domain: str,
    items: list,
    log_lines: list,
    gemini_model: str,
    prompt_style: str,
    thinking_budget: int,
    rate_limit_sleep: float,
    input_mode: str = "image",
    use_vertex: bool = False,
    warmup_contents: dict | None = None,
) -> dict:
    msg = f"\n=== Domain: {domain} ({len(items)} questions) ==="
    print(msg)
    log_lines.append(msg)

    answers = {}
    total_input_tokens = total_output_tokens = total_thinking_tokens = 0
    pbar = tqdm(items, desc=domain, unit="q")

    for item in pbar:
        t0 = time.time()
        raw = ""
        answer = "A"
        usage = None

        for attempt in range(8):
            try:
                parts: list[types.Part] = []

                # Few-shot examples as leading text (warmup 未使用時のみ)
                fewshot_text = ""
                if not warmup_contents:
                    for ex in item.get("fewshot_examples", []):
                        fewshot_text += f"{ex['prompt']}\nAnswer: {ex['answer']}\n\n"
                    if fewshot_text:
                        parts.append(types.Part.from_text(text=fewshot_text))

                # Images or Video
                video_path = None
                file_name = None
                if input_mode == "video" and len(item["images"]) > 1:
                    video_path = frames_to_video(item["images"])
                    if use_vertex:
                        parts.append(types.Part.from_bytes(
                            data=Path(video_path).read_bytes(), mime_type="video/mp4"))
                    else:
                        file_name, file_uri = upload_video(client, video_path)
                        parts.append(types.Part.from_uri(
                            file_uri=file_uri, mime_type="video/mp4"))
                else:
                    for img_path in item["images"]:
                        parts.append(load_image_part(img_path))

                # Question
                parts.append(types.Part.from_text(text=item["prompt"]))

                system_instruction = DOMAIN_SYSTEM.get(
                    item["domain"]) if prompt_style == "domain" else None

                effective_budget = thinking_budget
                if effective_budget == 0 and requires_thinking(gemini_model):
                    effective_budget = -1  # dynamic thinking
                thinking_cfg = types.ThinkingConfig(
                    thinking_budget=effective_budget) if effective_budget >= 0 else None
                max_tokens = 8192 if effective_budget != 0 else 24

                config = types.GenerateContentConfig(
                    temperature=0.0 if thinking_budget == 0 else 1.0,
                    max_output_tokens=max_tokens,
                    thinking_config=thinking_cfg,
                    system_instruction=system_instruction,
                )

                # Warm-up 会話を前置する (test 問題同士は干渉しないようコピー)
                if warmup_contents:
                    warmup_key = f"{item['domain']}::{item.get('question_type', '')}"
                    prefix = list(warmup_contents.get(warmup_key, []))
                    if not prefix and attempt == 0:
                        pbar.write(
                            f"  WARNING: no warmup found for key='{warmup_key}'")
                    contents = prefix + \
                        [types.Content(role="user", parts=parts)]
                else:
                    contents = [types.Content(role="user", parts=parts)]

                response = client.models.generate_content(
                    model=gemini_model,
                    contents=contents,
                    config=config,
                )

                raw = response.text.strip() if response.text else ""
                answer = extract_answer(raw)
                usage = response.usage_metadata
                break

            except Exception as e:
                wait = 2 ** attempt
                pbar.write(
                    f"  Error id={item['id']} attempt={attempt}: {e}  (retry in {wait}s)")
                time.sleep(wait)
            finally:
                # 一時動画ファイルとFiles APIエントリを削除
                if video_path and os.path.exists(video_path):
                    os.unlink(video_path)
                    video_path = None
                if file_name:
                    try:
                        client.files.delete(name=file_name)
                    except Exception:
                        pass
                    file_name = None

        # Token accounting
        if usage:
            in_tok = usage.prompt_token_count or 0
            out_tok = usage.candidates_token_count or 0
            thi_tok = getattr(usage, "thoughts_token_count", 0) or 0
            total_input_tokens += in_tok
            total_output_tokens += out_tok
            total_thinking_tokens += thi_tok
            total_cost = estimate_cost(
                gemini_model, total_input_tokens, total_output_tokens, total_thinking_tokens)
            cost_str = f" | tok={in_tok}in/{out_tok}out/{thi_tok}think  cumulative=${total_cost:.4f}"
        else:
            cost_str = ""

        gt = item["ground_truth"]
        correct = " ✓" if gt and answer == gt else (
            f" ✗(gt={gt})" if gt else "")
        answers[item["id"]] = answer

        elapsed = time.time() - t0
        snippet = raw[:60].replace("\n", " ")
        line = f"id={item['id']} raw='{snippet}' → {answer}{correct} ({elapsed:.1f}s){cost_str}"
        pbar.write(line)
        log_lines.append(line)

        time.sleep(rate_limit_sleep)

    domain_cost = estimate_cost(
        gemini_model, total_input_tokens, total_output_tokens, total_thinking_tokens)
    summary = (
        f"  [{domain}] tokens: {total_input_tokens}in / {total_output_tokens}out / {total_thinking_tokens}think"
        f"  estimated cost: ${domain_cost:.4f}"
    )
    print(summary)
    log_lines.append(summary)

    return answers, total_input_tokens, total_output_tokens, total_thinking_tokens


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
    parser.add_argument("--mode", choices=["test", "eval"], default="eval",
                        help="test: 提出用予測 / eval: サポートセットで正解率確認")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Geminiモデル名 (default: {DEFAULT_MODEL})")
    parser.add_argument("--prompt-style", choices=["default", "domain"], default="domain",
                        help="default: 末尾指示のみ / domain: ドメイン別システムプロンプト付き")
    parser.add_argument("--thinking-budget", type=int, default=0,
                        help="thinking token budget: 0=無効, -1=動的, N=最大N token (default: 0)")
    parser.add_argument("--input-mode", choices=["image", "video"], default="image",
                        help="image: フレームを個別画像として入力 / video: ffmpegでmp4化してFiles API経由で入力 (default: image)")
    parser.add_argument("--fewshot", action="store_true",
                        help="same-question few-shot examples をプロンプト先頭に付加")
    parser.add_argument("--rate-limit-sleep", type=float, default=0.5,
                        help="リクエスト間のsleep秒数 (default: 0.5)")
    parser.add_argument("--use-vertex", action="store_true",
                        help="Vertex AI を使用 (デフォルト: AI Studio API key)")
    parser.add_argument("--project", default="",
                        help="Vertex AI 使用時のGCPプロジェクトID (GOOGLE_CLOUD_PROJECT 環境変数でも可)")
    parser.add_argument("--location", default="global",
                        help="Vertex AI ロケーション (default: global)")
    parser.add_argument("--limit", type=int, default=0,
                        help="先頭N件だけ処理 (0=全件, default: 0)")
    parser.add_argument("--domain", default="",
                        help="特定ドメインのみ処理 (animal/industry/xsports/surgery, 空=全ドメイン)")
    parser.add_argument("--resume-from-id", default="",
                        help="指定IDのアイテムから末尾まで処理 (途中再開用)")
    parser.add_argument("--warmup-file", default="",
                        help="warmup_conversations.json のパス (指定時: warm-up 会話を前置して推論)")
    parser.add_argument("--warmup-max-frames", type=int, default=0,
                        help="warmup 各ターンの画像フレーム上限 (0=制限なし, default: 0)")
    args = parser.parse_args()

    # Client setup
    if args.use_vertex:
        vertex_api_key = os.environ.get("VERTEX_AI_API_KEY", "")
        if not vertex_api_key:
            raise ValueError("VERTEX_AI_API_KEY 環境変数が必要です")
        client = genai.Client(vertexai=True, api_key=vertex_api_key)
        print("Using Vertex AI (API key, global endpoint)")
    else:
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            raise ValueError("GEMINI_API_KEY 環境変数が必要です (または --use-vertex を指定)")
        client = genai.Client(api_key=api_key)
        print("Using Google AI Studio (API key)")

    OUTPUT_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_lines = [
        f"Run started: {timestamp}  mode={args.mode}  model={args.model}"
        f"  prompt_style={args.prompt_style}  thinking_budget={args.thinking_budget}"
        f"  fewshot={args.fewshot}  use_vertex={args.use_vertex}"
        f"  warmup_file={args.warmup_file}"
    ]

    warmup_contents = None
    if args.warmup_file:
        warmup_contents = load_warmup_contents(
            args.warmup_file, max_frames=args.warmup_max_frames)

    if args.mode == "test":
        items = load_test_items(fewshot=args.fewshot)
        with open(SUBMISSION_TEMPLATE) as f:
            submission = json.load(f)
    else:
        items = load_eval_items(fewshot=args.fewshot)

    if args.resume_from_id:
        ids = [str(it["id"]) for it in items]
        if args.resume_from_id in ids:
            start_idx = ids.index(args.resume_from_id)
            items = items[start_idx:]
            print(
                f"(--resume-from-id {args.resume_from_id}: {len(items)} items remaining)")
        else:
            print(
                f"WARNING: id={args.resume_from_id} not found, processing all items")

    if args.domain:
        items = [it for it in items if it["domain"] == args.domain]
        print(f"(--domain {args.domain}: {len(items)} items)")

    if args.limit > 0:
        items = items[:args.limit]
        print(f"(--limit {args.limit}: {len(items)} items)")

    by_domain = defaultdict(list)
    for item in items:
        by_domain[item["domain"]].append(item)

    all_answers = {}
    grand_in = grand_out = grand_think = 0
    for domain, domain_items in by_domain.items():
        domain_answers, in_tok, out_tok, think_tok = run_domain(
            client, domain, domain_items, log_lines,
            gemini_model=args.model,
            prompt_style=args.prompt_style,
            thinking_budget=args.thinking_budget,
            rate_limit_sleep=args.rate_limit_sleep,
            input_mode=args.input_mode,
            use_vertex=args.use_vertex,
            warmup_contents=warmup_contents,
        )
        all_answers.update(domain_answers)
        grand_in += in_tok
        grand_out += out_tok
        grand_think += think_tok

    total_cost = estimate_cost(args.model, grand_in, grand_out, grand_think)
    cost_line = (
        f"\n=== Total token usage ==="
        f"\n  input={grand_in}  output={grand_out}  thinking={grand_think}"
        f"\n  Estimated total cost: ${total_cost:.4f} (~¥{total_cost * 150:.0f})"
    )
    print(cost_line)
    log_lines.append(cost_line)

    if args.mode == "test":
        for entry in submission:
            entry["answer"] = all_answers.get(entry["id"], "A")
        pred_path = OUTPUT_DIR / f"predictions_gemini_{timestamp}.json"
        with open(pred_path, "w") as f:
            json.dump(submission, f, indent=2)
        line = f"\nSaved {len(submission)} predictions → {pred_path}"
        print(line)
        log_lines.append(line)
    else:
        print_accuracy(items, all_answers, log_lines)

    log_path = OUTPUT_DIR / f"log_gemini_{args.mode}_{timestamp}.txt"
    with open(log_path, "w") as f:
        f.write("\n".join(log_lines))
    print(f"Log → {log_path}")


if __name__ == "__main__":
    main()
