Data Preparation

1. Test Set (for Evaluation)
   The test set contains questions only (no ground-truth answers).

# Using huggingface-cli

pip install huggingface_hub
huggingface-cli download myuniverse/EgoCross --local-dir ./EgoCross_test --repo-type dataset

# Or using git

git lfs install
git clone https://huggingface.co/datasets/myuniverse/EgoCross
Link: https://huggingface.co/datasets/myuniverse/EgoCross

2. Support Set (for Training/Few-shot)
   The support set contains 80 samples with answers (20 per domain) for training or few-shot learning.

# Using modelscope

pip install modelscope
modelscope download --dataset YuLi2024/EgoCross_support_set --local_dir ./EgoCross_support
Link: https://modelscope.cn/datasets/YuLi2024/EgoCross_support_set

3. Pre-trained Models (Optional)
   We provide fine-tuned Qwen3-VL-4B models per domain:

Surgery: https://modelscope.cn/models/YuLi2024/EgoCross_sft_qwen3vl4B_surgery
Industry: https://modelscope.cn/models/YuLi2024/EgoCross_sft_qwen3vl4B_industry
XSports: https://modelscope.cn/models/YuLi2024/EgoCross_sft_qwen3vl4B_xsports
Animal: https://modelscope.cn/models/YuLi2024/EgoCross_sft_qwen3vl4B_animal
Example download:

modelscope download --model YuLi2024/EgoCross_sft_qwen3vl4B_surgery --local_dir ./models/surgery 4) Data Format
Test Set Format
{
"id": 1,
"dataset": "CholecTrack20",
"primary_category": "Counting",
"question_type": "distinct instruments counting",
"question_text": "How many distinct types of surgical instruments were visible?",
"options": [
"A: 4",
"B: 3",
"C: 1",
"D: 2"
],
"question_id": "CholecTrack20_VID01_q1_distinct-instruments-counting",
"video_path": [
"path/to/frame_00000.jpg",
"path/to/frame_00001.jpg"
]
}
Support Set Format (ShareGPT)
{
"messages": [
{
"role": "user",
"content": "<image><image>...<image>Question text?\nA) Option A\nB) Option B\nC) Option C\nD) Option D"
},
{
"role": "assistant",
"content": "A"
}
],
"images": ["path/to/frame1.jpg", "path/to/frame2.jpg"],
"domain": "surgery"
} 5) Quick Start (Optional)
Minimal inference example (vLLM-compatible OpenAI API):

from openai import OpenAI
import json
import base64

def encode_image(path):
with open(path, "rb") as f:
return base64.b64encode(f.read()).decode()

with open("EgoCross_test/test.json") as f:
questions = json.load(f)

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

predictions = {}
for q in questions:
content = []
for img_path in q["video_path"]:
content.append({
"type": "image_url",
"image_url": {"url": f"data:image/jpeg;base64,{encode_image(img_path)}"}
})

    prompt = q["question_text"] + "\n" + "\n".join(q["options"])
    content.append({"type": "text", "text": prompt})

    response = client.chat.completions.create(
        model="your_model",
        messages=[{"role": "user", "content": content}],
        max_tokens=16
    )

    answer = response.choices[0].message.content.strip().upper()[0]
    predictions[q["question_id"]] = answer

with open("predictions.json", "w") as f:
json.dump(predictions, f, indent=2)
