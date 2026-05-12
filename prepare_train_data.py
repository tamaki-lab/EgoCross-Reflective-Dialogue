"""
train.json の temporal 問題に [Frame at X.Xs] タイムスタンプを追加して
data/egocross/train_frame_ts.json を生成する。
"""
import json
import re

from common import BASE, SUPPORT_JSON, CLASSIFY_JSON, DOMAIN_ORIG_FPS, _compute_eval_timestamps

TRAIN_JSON = SUPPORT_JSON
OUTPUT_JSON = BASE / "data/egocross/train_frame_ts.json"


def main():
    with open(TRAIN_JSON) as f:
        data = json.load(f)

    qt_map: dict[int, str] = {}
    if CLASSIFY_JSON.exists():
        with open(CLASSIFY_JSON) as f:
            qt_map = {e["index"]: e["predicted_type"] for e in json.load(f)}

    result = []
    n_modified = 0
    for i, d in enumerate(data):
        qt = qt_map.get(i, "")
        if "temporal" in qt.lower() and d.get("images"):
            orig_fps = DOMAIN_ORIG_FPS.get(d.get("domain", ""), 30.0)
            timestamps = _compute_eval_timestamps(d["images"], orig_fps)

            content = d["messages"][0]["content"]
            parts = []
            pos = 0
            img_idx = 0
            for m in re.finditer(r"<image>", content):
                parts.append(content[pos:m.start()])
                if img_idx < len(timestamps):
                    parts.append(f"[Frame at {timestamps[img_idx]:.1f}s]")
                parts.append("<image>")
                img_idx += 1
                pos = m.end()
            parts.append(content[pos:])

            new_d = {**d, "messages": [
                {**d["messages"][0], "content": "".join(parts)},
                *d["messages"][1:],
            ]}
            result.append(new_d)
            n_modified += 1
        else:
            result.append(d)

    with open(OUTPUT_JSON, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Modified {n_modified}/{len(data)} items (temporal questions)")
    print(f"Saved → {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
