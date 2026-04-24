# EgoCross: Cross-Domain Egocentric Video Understanding

Fine-tuning Vision-Language Models on egocentric video understanding tasks across multiple domains.

---

## 🏆 EgoVis Challenge @ CVPR 2026

We are excited to announce that **EgoCross** is featured in the **EgoVis Challenge**!

Participate in our competition:

| Track              | Description                         | Link                                                                   |
| ------------------ | ----------------------------------- | ---------------------------------------------------------------------- |
| **Source-Limited** | Fine-tune with provided support set | [Codabench Competition](https://www.codabench.org/competitions/11279/) |
| **Source-Free**    | Zero-shot evaluation only           | [Codabench Competition](https://www.codabench.org/competitions/13868/) |

---

## 📦 Resources

### Support Set (with answers, for training)

- **ModelScope**: [EgoCross_support_set](https://modelscope.cn/datasets/YuLi2024/EgoCross_support_set)

### Test Set (questions only, no answers)

- **HuggingFace**: [EgoCross](https://huggingface.co/datasets/myuniverse/EgoCross)

> **Note**: The test set contains only questions without ground-truth answers. Submit predictions to Codabench for evaluation.

### Pre-trained Models (Full SFT, Epoch 2)

| Domain   | ModelScope Link                                                                                          |
| -------- | -------------------------------------------------------------------------------------------------------- |
| Animal   | [EgoCross_sft_qwen3vl4B_animal](https://modelscope.cn/models/YuLi2024/EgoCross_sft_qwen3vl4B_animal)     |
| Industry | [EgoCross_sft_qwen3vl4B_industry](https://modelscope.cn/models/YuLi2024/EgoCross_sft_qwen3vl4B_industry) |
| XSports  | [EgoCross_sft_qwen3vl4B_xsports](https://modelscope.cn/models/YuLi2024/EgoCross_sft_qwen3vl4B_xsports)   |
| Surgery  | [EgoCross_sft_qwen3vl4B_surgery](https://modelscope.cn/models/YuLi2024/EgoCross_sft_qwen3vl4B_surgery)   |

---

## 📊 Dataset Statistics

| Domain    | Source Dataset             | Support Samples | Test Questions | Description                    |
| --------- | -------------------------- | --------------- | -------------- | ------------------------------ |
| Animal    | EgoPet                     | 20              | 183            | Pet-mounted camera footage     |
| Industry  | ENIGMA                     | 20              | 245            | Industrial assembly operations |
| XSports   | ExtremeSportFPV            | 20              | 246            | First-person extreme sports    |
| Surgery   | CholecTrack20 + EgoSurgery | 20              | 283            | Laparoscopic surgery videos    |
| **Total** | -                          | **80**          | **957**        | -                              |

---

## 🚀 Quick Start

### Step 1: Install LLaMA-Factory

[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) is an easy-to-use framework for fine-tuning LLMs and VLMs.

```bash
# Clone LLaMA-Factory
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory

# Install with Qwen support
pip install -e ".[torch,metrics,qwen]"

# Install additional dependencies for Qwen3-VL
pip install transformers>=4.57.0 qwen-vl-utils
```

### Step 2: Download Dataset

```bash
# Install ModelScope CLI
pip install modelscope

# Download support set
modelscope download --dataset YuLi2024/EgoCross_support_set --local_dir ./data/egocross
```

### Step 3: Prepare Dataset Configuration

Create `data/dataset_info.json`:

```json
{
  "egocross_train": {
    "file_name": "egocross/train.json",
    "formatting": "sharegpt",
    "columns": {
      "messages": "messages",
      "images": "images"
    }
  }
}
```

**Important**: Update image paths in `train.json` to absolute paths:

```bash
# Example: Convert relative paths to absolute
sed -i 's|"frames/|"/absolute/path/to/data/egocross/frames/|g' data/egocross/train.json
```

### Step 4: Create Training Configuration

Create `configs/full_sft.yaml`:

```yaml
### Model
model_name_or_path: Qwen/Qwen3-VL-4B-Instruct
trust_remote_code: true

### Method
stage: sft
do_train: true
finetuning_type: full

### Dataset
dataset: egocross_train
dataset_dir: ./data
template: qwen3_vl
cutoff_len: 32768
preprocessing_num_workers: 16

### Image/Video Processing
image_max_pixels: 360000
video_max_pixels: 360000
image_min_pixels: 50176

### Training Hyperparameters
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1.0e-5
num_train_epochs: 2
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true

### Output
output_dir: ./output/egocross_full_sft
logging_steps: 10
save_steps: 500

### DeepSpeed (REQUIRED for Full SFT to avoid OOM)
deepspeed: examples/deepspeed/ds_z2_config.json
```

### Step 5: Run Training

```bash
cd LLaMA-Factory

# Full SFT requires 4 GPUs (A100/H100 recommended)
FORCE_TORCHRUN=1 NNODES=1 NPROC_PER_NODE=4 \
    llamafactory-cli train configs/full_sft.yaml
```

For single GPU, use LoRA instead (see [LoRA Configuration](#lora-configuration) below).

---

## ⚙️ Training Configurations

### Full SFT (Recommended)

| Parameter                     | Value                       | Notes                         |
| ----------------------------- | --------------------------- | ----------------------------- |
| `model_name_or_path`          | `Qwen/Qwen3-VL-4B-Instruct` | Base model                    |
| `finetuning_type`             | `full`                      | Full parameter tuning         |
| `learning_rate`               | `1e-5`                      | Lower LR for full fine-tuning |
| `per_device_train_batch_size` | `1`                         | Per GPU batch size            |
| `gradient_accumulation_steps` | `8`                         | Effective batch = 1×8×4 = 32  |
| `num_train_epochs`            | `2`                         | 1-2 epochs sufficient         |
| `image_max_pixels`            | `360000`                    | ~600×600, prevents OOM        |
| `deepspeed`                   | `ds_z2_config.json`         | **Required** to avoid OOM     |

**Hardware**: 4× A100/H100 GPUs, ~40GB VRAM per GPU

### LoRA Configuration

For single GPU training, create `configs/lora.yaml`:

```yaml
### Model
model_name_or_path: Qwen/Qwen3-VL-4B-Instruct
trust_remote_code: true

### Method
stage: sft
do_train: true
finetuning_type: lora

### LoRA Config
lora_rank: 64
lora_alpha: 128
lora_dropout: 0.05
lora_target: all

### Dataset
dataset: egocross_train
dataset_dir: ./data
template: qwen3_vl
cutoff_len: 32768

### Image/Video Processing
image_max_pixels: 360000
video_max_pixels: 360000
image_min_pixels: 50176

### Training Hyperparameters
per_device_train_batch_size: 2
gradient_accumulation_steps: 4
learning_rate: 2.0e-4
num_train_epochs: 2
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true

### Output
output_dir: ./output/egocross_lora
logging_steps: 10
save_steps: 500
```

Run LoRA training:

```bash
llamafactory-cli train configs/lora.yaml
```

**Hardware**: 1× A100/H100 GPU, ~24GB VRAM

### LoRA Merge (Required for Inference)

vLLM does not support LoRA adapters for VLMs directly. Merge first:

```bash
llamafactory-cli export \
    --model_name_or_path Qwen/Qwen3-VL-4B-Instruct \
    --adapter_name_or_path ./output/egocross_lora \
    --template qwen3_vl \
    --finetuning_type lora \
    --export_dir ./output/egocross_lora_merged \
    --export_size 4 \
    --export_device cuda \
    --export_legacy_format false
```

---

## 🔮 Inference

### Using vLLM

```bash
# Start vLLM server
python -m vllm.entrypoints.openai.api_server \
    --model ./output/egocross_full_sft \
    --port 8000 \
    --served-model-name egocross \
    --trust-remote-code
```

### Python Inference Example

```python
from openai import OpenAI
import base64

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

# Build message with multiple frames
content = []
image_paths = ["frame_00.jpg", "frame_10.jpg", "frame_20.jpg", "frame_30.jpg"]

for img_path in image_paths:
    content.append({
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{encode_image(img_path)}"}
    })

question = """What is the person doing in this video?
A) Cooking food
B) Washing dishes
C) Setting the table
D) Cleaning the counter"""

content.append({"type": "text", "text": question})

response = client.chat.completions.create(
    model="egocross",
    messages=[{"role": "user", "content": content}],
    max_tokens=16,
    temperature=0
)

print(response.choices[0].message.content)  # Output: "A"
```

---

## 📈 Results

Training on EgoCross benchmark with Qwen3-VL-4B:

| Method   | Epochs | Surgery | Industry | XSports | Animal | Overall   |
| -------- | ------ | ------- | -------- | ------- | ------ | --------- |
| Baseline | -      | 48.41   | 34.29    | 44.72   | 55.19  | 45.14     |
| Full SFT | 1      | 46.29   | 36.33    | 48.37   | 55.19  | 45.98     |
| Full SFT | 2      | 47.70   | 35.10    | 48.37   | 55.19  | **46.08** |

**Note:** Overall is micro-average (total correct / total questions = 957).

---

## 📝 Data Format

We use ShareGPT format for multi-modal training:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "<image><image><image><image>Based on these video frames, what is the person doing?\nA) Cooking food\nB) Washing dishes\nC) Setting the table\nD) Cleaning the counter"
    },
    {
      "role": "assistant",
      "content": "A"
    }
  ],
  "images": ["/absolute/path/to/frame_00.jpg", "/absolute/path/to/frame_10.jpg", "/absolute/path/to/frame_20.jpg", "/absolute/path/to/frame_30.jpg"]
}
```

**Key points:**

- Number of `<image>` tags must match length of `images` array
- Image paths must be **absolute paths**
- Answer should be a single letter (A, B, C, or D)

---

## ⚠️ Common Issues & Solutions

| Issue                      | Cause                      | Solution                                                        |
| -------------------------- | -------------------------- | --------------------------------------------------------------- |
| CUDA OOM during Full SFT   | Missing DeepSpeed          | Add `deepspeed: examples/deepspeed/ds_z2_config.json` to config |
| CUDA OOM with large images | image_max_pixels too high  | Set `image_max_pixels: 360000`                                  |
| vLLM tokenizer error       | LlamaFactory export bug    | Copy tokenizer files from base model (see below)                |
| Dataset not found          | Wrong dataset_dir          | Use absolute path in `dataset_dir`                              |
| `<image>` count mismatch   | Wrong number of image tags | Ensure `<image>` count equals `images` array length             |

### Tokenizer Fix

If vLLM fails with `AttributeError: 'list' object has no attribute 'keys'`:

```bash
# Copy tokenizer files from base model
cp /path/to/Qwen3-VL-4B-Instruct/tokenizer_config.json ./output/egocross_full_sft/
cp /path/to/Qwen3-VL-4B-Instruct/tokenizer.json ./output/egocross_full_sft/
```

---

## 📜 Citation

```bibtex
@article{li2025egocross,
  title={Egocross: Benchmarking multimodal large language models for cross-domain egocentric video question answering},
  author={Li, Yanjun and Fu, Yuqian and Qian, Tianwen and Xu, Qi'ao and Dai, Silong and Paudel, Danda Pani and Van Gool, Luc and Wang, Xiaoling},
  journal={arXiv preprint arXiv:2508.10729},
  year={2025}
}
```

---

## 🙏 Acknowledgments

- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) - Fine-tuning framework
- [Qwen3-VL](https://github.com/QwenLM/Qwen-VL) - Base vision-language model
- Original datasets: EgoPet, ENIGMA, ExtremeSportFPV, CholecTrack20, EgoSurgery

---

## 📧 Contact

For questions about the competition, please open an issue or contact the organizers through Codabench.
