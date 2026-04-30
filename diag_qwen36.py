"""Qwen3.6-27B の image_features 形状を確認する最小診断スクリプト"""
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from qwen_vl_utils import process_vision_info

MODEL = "Qwen/Qwen3.6-27B"
print(f"Loading {MODEL}...")
processor = AutoProcessor.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, device_map="auto")
model.eval()

# シンプルな1画像入力でテスト
import sys
img_path = sys.argv[1] if len(sys.argv) > 1 else None
if img_path is None:
    print("Usage: python diag_qwen36.py <image_path>")
    sys.exit(1)

messages = [{"role": "user", "content": [
    {"type": "image", "image": img_path},
    {"type": "text", "text": "What is shown? Answer A."},
]}]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
if isinstance(video_kwargs.get("fps"), list):
    video_kwargs["fps"] = video_kwargs["fps"][0] if video_kwargs["fps"] else None

inputs = processor(
    text=[text], images=image_inputs, videos=video_inputs,
    padding=True, return_tensors="pt", min_pixels=50176, max_pixels=50176,
    **video_kwargs,
).to(model.device)

print(f"input_ids shape: {inputs.input_ids.shape}")
print(f"pixel_values shape: {inputs.pixel_values.shape if 'pixel_values' in inputs else 'N/A'}")
print(f"image_grid_thw: {inputs.get('image_grid_thw')}")

# image_token_id を探す
image_token_id = model.config.image_token_id
print(f"image_token_id: {image_token_id}")
n_img_tokens = (inputs.input_ids == image_token_id).sum().item()
print(f"n_image_tokens in input_ids: {n_img_tokens}")

# get_image_features を直接呼ぶ
with torch.no_grad():
    img_out = model.model.get_image_features(
        inputs.pixel_values.to(model.dtype), inputs.image_grid_thw, return_dict=True)
    pooler = img_out.pooler_output
    print(f"pooler_output type: {type(pooler)}")
    if isinstance(pooler, (list, tuple)):
        print(f"  num items: {len(pooler)}")
        for i, p in enumerate(pooler):
            print(f"  [{i}] shape: {p.shape}, numel: {p.numel()}")
        cat = torch.cat(pooler, dim=0)
        print(f"after cat: shape={cat.shape}, numel={cat.numel()}")
    else:
        print(f"  shape: {pooler.shape}, numel: {pooler.numel()}")

    # text embed の hidden_size
    embed = model.get_input_embeddings()
    print(f"text embed hidden_size: {embed.embedding_dim}")
    print(f"vision merger output dim: {pooler[0].shape[-1] if isinstance(pooler, (list, tuple)) else pooler.shape[-1]}")
