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

### Pre-trained Models (LoRA SFT, Epoch 2)

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

Place `dataset_info.json` in `data/egocross/` (already included in this repo):

```json
{
  "egocross": {
    "file_name": "train.json",
    "formatting": "sharegpt",
    "columns": { "messages": "messages", "images": "images" },
    "tags": { "role_tag": "role", "content_tag": "content", "user_tag": "user", "assistant_tag": "assistant" }
  }
}
```

**Important**: Update image paths in `train.json` to absolute paths:

```bash
# Example: Convert relative paths to absolute
sed -i 's|"frames/|"/absolute/path/to/data/egocross/frames/|g' data/egocross/train.json
```

### Step 4: Run LoRA Training

```bash
# Standard training (single GPU)
bash bash/train.sh

# With frame timestamps (adds [Frame at X.Xs] to temporal questions)
bash bash/train.sh --frame-timestamps
```

Output: `output/egocross_lora_YYYYMMDD_HHMMSS/` (new directory per run, one checkpoint per epoch)

### Step 5: Run Inference

```bash
# Evaluate on support set
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 \
    python infer_all.py --mode eval \
    --adapter-path output/egocross_lora_YYYYMMDD_HHMMSS/checkpoint-80

# Generate test predictions
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 \
    python infer_all.py --mode test \
    --adapter-path output/egocross_lora_YYYYMMDD_HHMMSS/checkpoint-80
```

For the full pipeline including warmup, see [INFERENCE.md](../INFERENCE.md).

---

## ⚙️ Training Configuration

LoRA fine-tuning on a single GPU. Key settings in `configs/lora.yaml`:

| Parameter                     | Value                       | Description                         |
| ----------------------------- | --------------------------- | ----------------------------------- |
| `model_name_or_path`          | `Qwen/Qwen3-VL-4B-Instruct` | Base model                          |
| `finetuning_type`             | `lora`                      | LoRA (single GPU)                   |
| `template`                    | `qwen3_vl_nothink`          | Chat template (thinking disabled)   |
| `lora_rank`                   | `64`                        | LoRA rank                           |
| `image_max_pixels`            | `128000`                    | ~357×357, prevents OOM              |
| `learning_rate`               | `1.0e-5`                    | Learning rate                       |
| `num_train_epochs`            | `10`                        | Number of epochs                    |
| `per_device_train_batch_size` | `1`                         | Batch size per GPU                  |
| `gradient_accumulation_steps` | `1`                         | Gradient accumulation steps         |
| `save_strategy`               | `epoch`                     | Save a checkpoint after every epoch |

**Hardware**: 1× GPU with ~16GB+ VRAM

---

## 🔮 Inference

Adapters can be loaded directly — no merge required.

```bash
# Evaluate a specific checkpoint
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 \
    python infer_all.py --mode eval \
    --adapter-path output/egocross_lora_YYYYMMDD_HHMMSS/checkpoint-80

# Evaluate all checkpoints and compare
bash bash/eval_checkpoints.sh output/egocross_lora_YYYYMMDD_HHMMSS

# With warmup conversations prepended
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 \
    python infer_all.py --mode eval \
    --adapter-path output/egocross_lora_YYYYMMDD_HHMMSS/checkpoint-80 \
    --warmup-file outputs/warmup_conversations_qwen.json
```

See [INFERENCE.md](../INFERENCE.md) for the complete pipeline.

---

## 📈 Results

Eval accuracy on EgoCross support set with Qwen3-VL-4B:

| Method   | Epochs | Surgery | Industry | XSports | Animal | Overall   |
| -------- | ------ | ------- | -------- | ------- | ------ | --------- |
| Baseline | -      | 48.41   | 34.29    | 44.72   | 55.19  | 45.14     |
| LoRA SFT | 1      | 46.29   | 36.33    | 48.37   | 55.19  | 45.98     |
| LoRA SFT | 2      | 47.70   | 35.10    | 48.37   | 55.19  | **46.08** |

**Note:** Overall is micro-average (total correct / total questions = 957). Numbers are eval accuracy on the support set.

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

| Issue                      | Cause                     | Solution                                             |
| -------------------------- | ------------------------- | ---------------------------------------------------- |
| CUDA OOM during training   | image_max_pixels too high | Set `image_max_pixels: 128000` in `configs/lora.yaml` |
| Dataset not found          | Wrong dataset_dir         | Use path relative to repo root (e.g. `./data/egocross`) |
| `<image>` count mismatch   | Wrong number of image tags | Ensure `<image>` count equals `images` array length  |
| Wrong answers (all "A")    | Thinking mode active      | Use `template: qwen3_vl_nothink` or pass `--thinking` |

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
