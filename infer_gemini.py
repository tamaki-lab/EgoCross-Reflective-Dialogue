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
SUBMISSION_TEMPLATE = BASE / "submission_template.json"
IMAGE_BASE = BASE / "EgoCross_test"
OUTPUT_DIR = BASE / "outputs"

_DOMAIN_FORMAT = (
    " Output only: 'Final Answer: X' where X is A, B, C, or D. No explanation."
)

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
# cache_read: context cache read price (typically 25% of input)
# cache_storage: context cache storage price per 1M tokens per hour
PRICING = {
    "gemini-3.1-flash-image-preview": {"input": 0.50, "output": 3.00,  "thinking": 0.0,  "cache_read": 0.125,   "cache_storage": 1.00},
    "gemini-3.1-pro-preview":         {"input": 2.00, "output": 12.00, "thinking": 0.0,  "cache_read": 0.50,    "cache_storage": 4.50},
    "gemini-3-pro-preview":           {"input": 2.00, "output": 12.00, "thinking": 0.0,  "cache_read": 0.50,    "cache_storage": 4.50},
    "gemini-2.5-flash":               {"input": 0.075, "output": 0.30, "thinking": 3.50, "cache_read": 0.01875, "cache_storage": 1.00},
    "gemini-2.5-pro":                 {"input": 1.25,  "output": 10.00, "thinking": 3.50, "cache_read": 0.3125,  "cache_storage": 4.50},
    "gemini-2.5-flash-lite":          {"input": 0.01,  "output": 0.04,  "thinking": 3.50, "cache_read": 0.0025,  "cache_storage": 0.25},
}


def estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    thinking_tokens: int,
    cached_tokens: int = 0,
    cache_storage_token_hours: float = 0.0,
) -> float:
    p = PRICING.get(model, PRICING["gemini-2.5-flash"])
    non_cached = max(0, input_tokens - cached_tokens)
    return (
        non_cached / 1_000_000 * p["input"] +
        cached_tokens / 1_000_000 * p.get("cache_read", p["input"] * 0.25) +
        output_tokens / 1_000_000 * p["output"] +
        thinking_tokens / 1_000_000 * p["thinking"] +
        cache_storage_token_hours / 1_000_000 * p.get("cache_storage", 1.00)
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


def load_warmup_contents(path: str, max_frames: int = 0, input_mode: str = "image") -> dict[str, list[types.Content]]:
    """warmup_conversations.json を読み込み、画像バイトを復元して Content リストを返す。
    key は 'domain::question_type' 形式。
    max_frames > 0 のとき、各ユーザーターンの画像を先頭 max_frames 枚に制限する。
    input_mode == 'video' のとき、複数フレームをmp4に変換してインラインバイトで渡す。"""
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
        contents = [
            types.Content(role="user", parts=[
                          types.Part.from_text(text=preamble)]),
            types.Content(role="model", parts=[types.Part.from_text(
                text="Understood. I will study these practice questions and reflections carefully.")]),
        ]
        for turn in group["turns"]:
            if turn["role"] == "user":
                images = turn.get("images", [])
                if max_frames > 0 and len(images) > max_frames:
                    step = (len(images) - 1) / \
                        (max_frames - 1) if max_frames > 1 else 0
                    images = [images[round(i * step)]
                              for i in range(max_frames)]
                timestamps = turn.get("timestamps")
                if input_mode == "video" and len(images) > 1:
                    video_path = frames_to_video(images)
                    parts = [types.Part.from_bytes(
                        data=Path(video_path).read_bytes(), mime_type="video/mp4")]
                    os.unlink(video_path)
                elif timestamps and images:
                    parts = []
                    for i, p in enumerate(images):
                        parts.append(types.Part.from_text(text=f"[Frame at {timestamps[i]:.1f}s]"))
                        parts.append(load_image_part(p))
                else:
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
    mode_info = f", input_mode={input_mode}"
    print(
        f"Loaded warmup for {len(result)} groups from {path}{frame_info}{mode_info}")
    return result


def upload_file_to_api(client, path: str) -> tuple[str, str]:
    """単一画像ファイルをFiles APIにアップロードし(file_name, uri)を返す。"""
    p = Path(path)
    mime = MIME_MAP.get(p.suffix.lower(), "image/jpeg")
    with open(path, "rb") as f:
        uploaded = client.files.upload(
            file=f,
            config=types.UploadFileConfig(mime_type=mime),
        )
    while True:
        info = client.files.get(name=uploaded.name)
        if info.state.name != "PROCESSING":
            break
        time.sleep(0.5)
    return uploaded.name, info.uri


def load_warmup_with_files_api(
    client, path: str, max_frames: int = 0, input_mode: str = "image"
) -> tuple[dict[str, list[types.Content]], list[str]]:
    """warmup_conversations.json を読み込み、画像(またはmp4)を Files API にアップロードして
    URI ベースの Content リストとアップロード済み file_names リストを返す。
    Context Caching と組み合わせることで warmup input コストを ~75% 削減できる。
    input_mode == 'video' のとき、各ターンのフレームをmp4化してアップロードする。"""
    with open(path) as f:
        data = json.load(f)

    def apply_max_frames(images: list[str]) -> list[str]:
        if max_frames > 0 and len(images) > max_frames:
            step = (len(images) - 1) / \
                (max_frames - 1) if max_frames > 1 else 0
            return [images[round(i * step)] for i in range(max_frames)]
        return images

    all_file_names: list[str] = []
    result: dict[str, list[types.Content]] = {}

    if input_mode == "video":
        # 動画モード: ターンごとにmp4化してアップロード（重複排除不可）
        total_turns = sum(
            sum(1 for t in g["turns"] if t["role"]
                == "user" and t.get("images"))
            for g in data.values()
        )
        print(
            f"Converting and uploading {total_turns} warmup turns as video to Files API...")
        with tqdm(total=total_turns, desc="Uploading warmup videos") as pbar:
            for key, group in data.items():
                domain, qt = key.split("::", 1)
                preamble = (
                    f"The following are warm-up practice questions of type '{qt}' "
                    f"in the '{domain}' domain. "
                    "After each question you will see whether the answer was correct and, "
                    "if incorrect, a reflection on the mistake. "
                    "Study these carefully to improve your performance on similar questions."
                )
                contents = [
                    types.Content(role="user", parts=[
                                  types.Part.from_text(text=preamble)]),
                    types.Content(role="model", parts=[types.Part.from_text(
                        text="Understood. I will study these practice questions and reflections carefully.")]),
                ]
                for turn in group["turns"]:
                    if turn["role"] == "user":
                        images = apply_max_frames(turn.get("images", []))
                        parts = []
                        if len(images) > 1:
                            video_path = frames_to_video(images)
                            file_name, file_uri = upload_video(
                                client, video_path)
                            os.unlink(video_path)
                            all_file_names.append(file_name)
                            parts.append(types.Part.from_uri(
                                file_uri=file_uri, mime_type="video/mp4"))
                        else:
                            for img_path in images:
                                file_name, uri = upload_file_to_api(
                                    client, img_path)
                                all_file_names.append(file_name)
                                mime = MIME_MAP.get(
                                    Path(img_path).suffix.lower(), "image/jpeg")
                                parts.append(types.Part.from_uri(
                                    file_uri=uri, mime_type=mime))
                        if turn.get("text"):
                            parts.append(
                                types.Part.from_text(text=turn["text"]))
                        contents.append(types.Content(
                            role="user", parts=parts))
                        pbar.update(1)
                    else:
                        contents.append(types.Content(
                            role="model",
                            parts=[types.Part.from_text(text=turn["text"])],
                        ))
                result[key] = contents
    else:
        # 画像モード: ユニーク画像を一括アップロードして再利用
        all_image_paths: set[str] = set()
        for group in data.values():
            for turn in group["turns"]:
                if turn["role"] == "user":
                    all_image_paths.update(
                        apply_max_frames(turn.get("images", [])))

        print(
            f"Uploading {len(all_image_paths)} warmup images to Files API...")
        path_to_file: dict[str, tuple[str, str]] = {}
        for img_path in tqdm(sorted(all_image_paths), desc="Uploading warmup images"):
            file_name, uri = upload_file_to_api(client, img_path)
            path_to_file[img_path] = (file_name, uri)

        for key, group in data.items():
            domain, qt = key.split("::", 1)
            preamble = (
                f"The following are warm-up practice questions of type '{qt}' "
                f"in the '{domain}' domain. "
                "After each question you will see whether the answer was correct and, "
                "if incorrect, a reflection on the mistake. "
                "Study these carefully to improve your performance on similar questions."
            )
            contents = [
                types.Content(role="user", parts=[
                              types.Part.from_text(text=preamble)]),
                types.Content(role="model", parts=[types.Part.from_text(
                    text="Understood. I will study these practice questions and reflections carefully.")]),
            ]
            for turn in group["turns"]:
                if turn["role"] == "user":
                    images = apply_max_frames(turn.get("images", []))
                    parts = []
                    for img_path in images:
                        _, uri = path_to_file[img_path]
                        mime = MIME_MAP.get(
                            Path(img_path).suffix.lower(), "image/jpeg")
                        parts.append(types.Part.from_uri(
                            file_uri=uri, mime_type=mime))
                    if turn.get("text"):
                        parts.append(types.Part.from_text(text=turn["text"]))
                    contents.append(types.Content(role="user", parts=parts))
                else:
                    contents.append(types.Content(
                        role="model",
                        parts=[types.Part.from_text(text=turn["text"])],
                    ))
            result[key] = contents
        all_file_names = [fn for fn, _ in path_to_file.values()]

    frame_info = f", max_frames={max_frames}" if max_frames > 0 else ""
    print(
        f"Loaded warmup for {len(result)} groups from {path}{frame_info}, input_mode={input_mode}, uploaded {len(all_file_names)} files")
    return result, all_file_names


def _subsample_frames(images: list[str], max_frames: int) -> list[str]:
    if max_frames <= 0 or len(images) <= max_frames:
        return images
    step = (len(images) - 1) / (max_frames - 1) if max_frames > 1 else 0
    return [images[round(i * step)] for i in range(max_frames)]


DOMAIN_ORIG_FPS = {"surgery": 25.0, "industry": 30.0, "xsports": 30.0, "animal": 30.0}

TEST_SAMPLING_INTERVAL = 2.0  # 0.5 fps


def _compute_eval_timestamps(frame_paths: list[str], orig_fps: float) -> list[float]:
    """eval用: フレーム名の元インデックスからclip相対タイムスタンプを計算する。"""
    try:
        nums = [int(re.findall(r"\d+", Path(p).stem)[-1]) for p in frame_paths]
        min_n = min(nums)
        return [(n - min_n) / orig_fps for n in nums]
    except Exception:
        return [i * TEST_SAMPLING_INTERVAL for i in range(len(frame_paths))]


def _compute_test_timestamps(n_frames: int, interval: float = TEST_SAMPLING_INTERVAL) -> list[float]:
    """test用: サンプリング間隔からclip相対タイムスタンプを計算する。"""
    return [i * interval for i in range(n_frames)]


def _test_sampling_interval(dataset: str, video_paths: list[str]) -> float:
    """EgoSurgery と CholecTrack20 VID25/VID111 は 1fps (1.0s/frame)、それ以外は 0.5fps (2.0s)。"""
    if dataset == "EgoSurgery":
        return 1.0
    if dataset == "CholecTrack20":
        joined = "/".join(video_paths)
        if "VID25" in joined or "VID111" in joined:
            return 1.0
    return TEST_SAMPLING_INTERVAL


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


def load_test_items(fewshot: bool = False, visual_fewshot: bool = False) -> list[dict]:
    train_bank: dict[str, list] = {}
    if fewshot or visual_fewshot:
        with open(SUPPORT_JSON) as f:
            train_raw = json.load(f)
        qt_map_train = _load_support_question_types() if visual_fewshot else {}
        for idx, d in enumerate(train_raw):
            user_text = re.sub(
                r"<image>", "", d["messages"][0]["content"]).strip()
            entry = {
                "prompt": user_text,
                "answer": d["messages"][-1]["content"].strip().upper(),
                "images": d["images"],
            }
            if visual_fewshot:
                bank_key = f"{d['domain']}::{qt_map_train.get(idx, '')}"
            else:
                bank_key = user_text.split("\nA.")[0].strip()
            train_bank.setdefault(bank_key, []).append(entry)

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
        if visual_fewshot:
            lookup_key = f"{model_name}::{d.get('question_type', '')}"
        else:
            lookup_key = d["question_text"].strip()
        images = [str(IMAGE_BASE / p.lstrip("/")) for p in d["video_path"]]
        interval = _test_sampling_interval(d["dataset"], d["video_path"])
        items.append({
            "id": d["id"],
            "images": images,
            "timestamps": _compute_test_timestamps(len(images), interval),
            "prompt": prompt,
            "domain": model_name,
            "question_type": d.get("question_type", ""),
            "ground_truth": None,
            "fewshot_examples": train_bank.get(lookup_key, []),
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


def load_eval_items(fewshot: bool = False, visual_fewshot: bool = False) -> list[dict]:
    with open(SUPPORT_JSON) as f:
        raw = json.load(f)

    qt_map = _load_support_question_types()

    if visual_fewshot:
        # 全件評価 + warmupと同じ domain::question_type 単位で leave-one-out few-shot bank を構築
        by_group: dict[str, list] = defaultdict(list)
        for i, d in enumerate(raw):
            user_text = re.sub(r"<image>", "", d["messages"][0]["content"]).strip()
            group_key = f"{d['domain']}::{qt_map.get(i, '')}"
            by_group[group_key].append((i, d, user_text))

        items = []
        for group_key, group in by_group.items():
            group_examples = [
                {
                    "prompt": user_text,
                    "answer": d["messages"][-1]["content"].strip().upper(),
                    "images": d["images"],
                }
                for (_, d, user_text) in group
            ]
            for idx, (i, d, user_text) in enumerate(group):
                examples = [ex for j, ex in enumerate(group_examples) if j != idx]
                prompt = user_text + "\nYou MUST pick exactly one option. End your response with a single line: 'Final Answer: X' where X is one of A, B, C, or D."
                items.append({
                    "id": i,
                    "images": d["images"],
                    "timestamps": _compute_eval_timestamps(d["images"], DOMAIN_ORIG_FPS.get(d["domain"], 30.0)),
                    "prompt": prompt,
                    "domain": d["domain"],
                    "question_type": qt_map.get(i, ""),
                    "ground_truth": d["messages"][-1]["content"].strip().upper(),
                    "fewshot_examples": examples,
                })
        return items

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
                    "images": d["images"],
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
                "timestamps": _compute_eval_timestamps(d["images"], DOMAIN_ORIG_FPS.get(d["domain"], 30.0)),
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
            "timestamps": _compute_eval_timestamps(d["images"], DOMAIN_ORIG_FPS.get(d["domain"], 30.0)),
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
    visual_fewshot: bool = False,
    visual_fewshot_max_frames: int = 0,
    frame_timestamps: bool = False,
) -> dict:
    msg = f"\n=== Domain: {domain} ({len(items)} questions) ==="
    print(msg)
    log_lines.append(msg)

    answers = {}
    total_input_tokens = total_output_tokens = total_thinking_tokens = 0
    total_cached_tokens = 0
    total_storage_token_hours = 0.0
    pbar = tqdm(items, desc=domain, unit="q")

    # warmup 使用時は question_type 単位でグループ化してキャッシュを共有する
    if warmup_contents:
        by_qt: dict[str, list] = defaultdict(list)
        for item in items:
            by_qt[item.get("question_type", "")].append(item)
        qt_groups = list(by_qt.items())
    else:
        qt_groups = [("", items)]

    for qt, qt_items in qt_groups:
        # question_type グループごとに CachedContent を作成
        cache = None
        if warmup_contents and qt:
            warmup_key = f"{domain}::{qt}"
            prefix = warmup_contents.get(warmup_key, [])
            if prefix:
                system_instruction = build_warmup_system(domain, qt)
                try:
                    cache = client.caches.create(
                        model=gemini_model,
                        config=types.CreateCachedContentConfig(
                            system_instruction=system_instruction,
                            contents=prefix,
                            ttl="10800s",
                        ),
                    )
                    cache_token_count = getattr(
                        getattr(cache, "usage_metadata", None), "total_token_count", 0) or 0
                    cache_start_time = time.time()
                    pbar.write(
                        f"  Created cache for '{warmup_key}': {cache.name} ({cache_token_count} tokens)")
                except Exception as e:
                    pbar.write(
                        f"  Cache creation failed for '{warmup_key}': {e} — falling back to inline")
            else:
                pbar.write(
                    f"  WARNING: no warmup found for key='{warmup_key}'")

        try:
            for item in qt_items:
                t0 = time.time()
                raw = ""
                answer = "A"
                usage = None

                for attempt in range(8):
                    try:
                        parts: list[types.Part] = []

                        # Few-shot examples as leading text (warmup/visual_fewshot 未使用時のみ)
                        if not warmup_contents and not visual_fewshot:
                            fewshot_text = ""
                            for ex in item.get("fewshot_examples", []):
                                fewshot_text += f"{ex['prompt']}\nAnswer: {ex['answer']}\n\n"
                            if fewshot_text:
                                parts.append(
                                    types.Part.from_text(text=fewshot_text))

                        # Images or Video
                        video_path = None
                        file_name = None
                        if input_mode == "video" and len(item["images"]) > 1:
                            video_path = frames_to_video(item["images"])
                            if use_vertex:
                                parts.append(types.Part.from_bytes(
                                    data=Path(video_path).read_bytes(), mime_type="video/mp4"))
                            else:
                                file_name, file_uri = upload_video(
                                    client, video_path)
                                parts.append(types.Part.from_uri(
                                    file_uri=file_uri, mime_type="video/mp4"))
                        elif frame_timestamps and item.get("timestamps") and "temporal" in item.get("question_type", "").lower():
                            ts = item["timestamps"]
                            for i, img_path in enumerate(item["images"]):
                                if i < len(ts):
                                    parts.append(types.Part.from_text(text=f"[Frame at {ts[i]:.1f}s]"))
                                parts.append(load_image_part(img_path))
                        else:
                            for img_path in item["images"]:
                                parts.append(load_image_part(img_path))

                        # Question
                        parts.append(types.Part.from_text(text=item["prompt"]))

                        effective_budget = thinking_budget
                        if effective_budget == 0 and requires_thinking(gemini_model):
                            effective_budget = -1  # dynamic thinking
                        thinking_cfg = types.ThinkingConfig(
                            thinking_budget=effective_budget) if effective_budget >= 0 else None
                        max_tokens = 8192 if effective_budget != 0 else 24

                        if cache:
                            # system_instruction はキャッシュに含まれるので指定不要
                            separator = types.Part.from_text(
                                text="Warm-up complete. Now answer the following question:")
                            contents = [types.Content(
                                role="user", parts=[separator] + parts)]
                            config = types.GenerateContentConfig(
                                cached_content=cache.name,
                                temperature=0.0 if thinking_budget == 0 else 1.0,
                                max_output_tokens=max_tokens,
                                thinking_config=thinking_cfg,
                            )
                        elif warmup_contents:
                            # キャッシュ作成失敗時のインラインフォールバック
                            wk = f"{domain}::{item.get('question_type', '')}"
                            prefix_fb = list(warmup_contents.get(wk, []))
                            si = build_warmup_system(domain, item.get(
                                "question_type", "")) if prefix_fb else DOMAIN_SYSTEM.get(domain)
                            separator = types.Part.from_text(
                                text="Warm-up complete. Now answer the following question:")
                            contents = prefix_fb + [
                                types.Content(role="user", parts=[separator] + parts)]
                            config = types.GenerateContentConfig(
                                temperature=0.0 if thinking_budget == 0 else 1.0,
                                max_output_tokens=max_tokens,
                                thinking_config=thinking_cfg,
                                system_instruction=si,
                            )
                        else:
                            system_instruction = DOMAIN_SYSTEM.get(
                                item["domain"]) if prompt_style == "domain" else None
                            if visual_fewshot and item.get("fewshot_examples"):
                                vf_contents = []
                                for ex in item["fewshot_examples"]:
                                    ex_imgs = _subsample_frames(
                                        ex.get("images", []), visual_fewshot_max_frames)
                                    ex_parts = [load_image_part(p) for p in ex_imgs]
                                    ex_parts.append(types.Part.from_text(text=ex["prompt"]))
                                    vf_contents.append(types.Content(role="user", parts=ex_parts))
                                    vf_contents.append(types.Content(role="model", parts=[
                                        types.Part.from_text(text=f"Final Answer: {ex['answer']}")
                                    ]))
                                contents = vf_contents + [types.Content(role="user", parts=parts)]
                            else:
                                contents = [types.Content(role="user", parts=parts)]
                            config = types.GenerateContentConfig(
                                temperature=0.0 if thinking_budget == 0 else 1.0,
                                max_output_tokens=max_tokens,
                                thinking_config=thinking_cfg,
                                system_instruction=system_instruction,
                            )

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
                    cached_tok = getattr(
                        usage, "cached_content_token_count", 0) or 0
                    total_input_tokens += in_tok
                    total_output_tokens += out_tok
                    total_thinking_tokens += thi_tok
                    total_cached_tokens += cached_tok
                    total_cost = estimate_cost(
                        gemini_model, total_input_tokens, total_output_tokens,
                        total_thinking_tokens, total_cached_tokens, total_storage_token_hours)
                    cache_str = f"/{cached_tok}cached" if cached_tok else ""
                    cost_str = f" | tok={in_tok}in{cache_str}/{out_tok}out/{thi_tok}think  cumulative=${total_cost:.4f}"
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

                pbar.update(1)
                time.sleep(rate_limit_sleep)

        finally:
            # グループ処理完了後にキャッシュを削除、ストレージコストを確定
            if cache:
                elapsed_hours = (time.time() - cache_start_time) / 3600
                total_storage_token_hours += cache_token_count * elapsed_hours
                try:
                    client.caches.delete(name=cache.name)
                    pbar.write(
                        f"  Deleted cache: {cache.name} (存在時間: {elapsed_hours*60:.1f}分)")
                except Exception:
                    pass

    domain_cost = estimate_cost(
        gemini_model, total_input_tokens, total_output_tokens,
        total_thinking_tokens, total_cached_tokens, total_storage_token_hours)
    summary = (
        f"  [{domain}] tokens: {total_input_tokens}in ({total_cached_tokens}cached)"
        f" / {total_output_tokens}out / {total_thinking_tokens}think"
        f"  estimated cost: ${domain_cost:.4f}"
    )
    print(summary)
    log_lines.append(summary)

    return answers, total_input_tokens, total_output_tokens, total_thinking_tokens, total_cached_tokens, total_storage_token_hours


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
    parser.add_argument("--visual-fewshot", action="store_true",
                        help="support setの画像+問題+回答をそのままvisual few-shotとして渡す (reflection/thinkingなし)")
    parser.add_argument("--visual-fewshot-max-frames", type=int, default=0,
                        help="visual few-shot 各例の画像フレーム上限 (0=制限なし, default: 0)")
    parser.add_argument("--frame-timestamps", action="store_true",
                        help="各フレームの時刻 '[Frame at X.Xs]' をプロンプトに追加する")
    args = parser.parse_args()

    # Client setup
    if args.use_vertex:
        project = args.project or os.environ.get("GOOGLE_CLOUD_PROJECT", "")
        if not project:
            raise ValueError("--project または GOOGLE_CLOUD_PROJECT 環境変数が必要です")
        client = genai.Client(
            vertexai=True, project=project, location=args.location)
        print(
            f"Using Vertex AI (ADC, project={project}, location={args.location})")
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
        f"  warmup_file={args.warmup_file}  frame_timestamps={args.frame_timestamps}"
    ]

    warmup_contents = None
    warmup_file_names: list[str] = []
    if args.warmup_file:
        if args.use_vertex:
            # Vertex AI は Files API 非対応 — インラインバイトで読み込む
            warmup_contents = load_warmup_contents(
                args.warmup_file, max_frames=args.warmup_max_frames, input_mode=args.input_mode)
        else:
            warmup_contents, warmup_file_names = load_warmup_with_files_api(
                client, args.warmup_file, max_frames=args.warmup_max_frames, input_mode=args.input_mode)

    if args.mode == "test":
        items = load_test_items(fewshot=args.fewshot, visual_fewshot=args.visual_fewshot)
        with open(SUBMISSION_TEMPLATE) as f:
            submission = json.load(f)
    else:
        items = load_eval_items(fewshot=args.fewshot, visual_fewshot=args.visual_fewshot)

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
    grand_in = grand_out = grand_think = grand_cached = 0
    grand_storage_token_hours = 0.0
    for domain, domain_items in by_domain.items():
        domain_answers, in_tok, out_tok, think_tok, cached_tok, storage_th = run_domain(
            client, domain, domain_items, log_lines,
            gemini_model=args.model,
            prompt_style=args.prompt_style,
            thinking_budget=args.thinking_budget,
            rate_limit_sleep=args.rate_limit_sleep,
            input_mode=args.input_mode,
            use_vertex=args.use_vertex,
            warmup_contents=warmup_contents,
            visual_fewshot=args.visual_fewshot,
            visual_fewshot_max_frames=args.visual_fewshot_max_frames,
            frame_timestamps=args.frame_timestamps,
        )
        all_answers.update(domain_answers)
        grand_in += in_tok
        grand_out += out_tok
        grand_think += think_tok
        grand_cached += cached_tok
        grand_storage_token_hours += storage_th

    total_cost = estimate_cost(args.model, grand_in, grand_out,
                               grand_think, grand_cached, grand_storage_token_hours)
    cost_line = (
        f"\n=== Total token usage ==="
        f"\n  input={grand_in} (cached={grand_cached})  output={grand_out}  thinking={grand_think}"
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

    # warmup でアップロードした Files API ファイルを削除
    if warmup_file_names:
        print(
            f"Cleaning up {len(warmup_file_names)} uploaded Files API files...")
        for fn in warmup_file_names:
            try:
                client.files.delete(name=fn)
            except Exception:
                pass
        print("Cleanup done.")


if __name__ == "__main__":
    main()
