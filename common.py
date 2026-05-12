import re
from pathlib import Path

BASE = Path(__file__).parent

# Paths
TEST_JSON = BASE / "EgoCross_test/egocross_testbed/egocross_testbed_imgs.json"
SUPPORT_JSON = BASE / "data/egocross/train.json"
SUBMISSION_TEMPLATE = BASE / "submission_template.json"
IMAGE_BASE = BASE / "EgoCross_test"
OUTPUT_DIR = BASE / "outputs"
CLASSIFY_JSON = BASE / "outputs/support_question_types.json"
MODEL_BASE = BASE / "models"

# Domain constants
DOMAIN_ORIG_FPS: dict[str, float] = {
    "surgery": 25.0, "industry": 30.0, "xsports": 30.0, "animal": 30.0,
}

_DOMAIN_BASE: dict[str, str] = {
    "animal":   "You are an expert analyzing egocentric video frames featuring animals. Carefully observe the animal species and behaviors shown.",
    "industry": "You are an expert analyzing egocentric video frames from industrial or factory settings. Carefully observe the tools, machinery, and work activities shown.",
    "xsports":  "You are an expert analyzing egocentric video frames from extreme sports. Carefully observe the sport type, actions, and environment shown.",
    "surgery":  "You are an expert analyzing egocentric video frames from surgical procedures. Carefully observe the instruments, tissues, and surgical actions shown.",
}

# test set: dataset name → domain name
DATASET_MODEL: dict[str, str] = {
    "CholecTrack20":   "surgery",
    "EgoSurgery":      "surgery",
    "ENIGMA":          "industry",
    "ExtrameSportFPV": "xsports",
    "EgoPet":          "animal",
}

TEST_SAMPLING_INTERVAL = 2.0  # 0.5 fps

# Gemini image MIME types
MIME_MAP: dict[str, str] = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".png": "image/png", ".gif": "image/gif", ".webp": "image/webp",
}


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


def _subsample_frames(images: list[str], max_frames: int) -> list[str]:
    if max_frames <= 0 or len(images) <= max_frames:
        return images
    step = (len(images) - 1) / (max_frames - 1) if max_frames > 1 else 0
    return [images[round(i * step)] for i in range(max_frames)]


def _load_support_question_types() -> dict[int, str]:
    """support_question_types.json が存在すれば index→question_type を返す。"""
    if not CLASSIFY_JSON.exists():
        return {}
    with open(CLASSIFY_JSON) as f:
        import json
        data = json.load(f)
    return {entry["index"]: entry["predicted_type"] for entry in data}


def build_warmup_system(domain: str, question_type: str, domain_format: str) -> str:
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
        + domain_format
    )


def load_image_part(path: str):
    """Gemini用: 画像ファイルを types.Part として読み込む。"""
    from google.genai import types
    p = Path(path)
    mime = MIME_MAP.get(p.suffix.lower(), "image/jpeg")
    return types.Part.from_bytes(data=p.read_bytes(), mime_type=mime)


def requires_thinking(model: str) -> bool:
    """Gemini 3.x系はthinking必須 (budget=0はAPI拒否)。"""
    return model.startswith("gemini-3")
