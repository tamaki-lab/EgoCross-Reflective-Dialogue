# EgoCross Inference & Warmup Guide

## Scripts

| Script                     | Role                                                     |
| -------------------------- | -------------------------------------------------------- |
| `classify_support.py`      | Classify support set questions by question type (Step 1) |
| `warmup_gemini.py`         | Generate warmup conversations with Gemini (Step 2a)      |
| `warmup_qwen.py`           | Generate warmup conversations with local Qwen (Step 2b)  |
| `infer_gemini.py`          | Run inference with Gemini                                |
| `infer_all.py`             | Run inference with local Qwen                            |
| `bash/train.sh`            | LoRA fine-tuning on the support set                      |
| `bash/eval_checkpoints.sh` | Evaluate all checkpoints and compare accuracy            |

---

## Prerequisites

Set API keys in `.env`:

```
GEMINI_API_KEY=...
GOOGLE_CLOUD_PROJECT=...
```

---

## LoRA Fine-tuning Flow

Use this flow when fine-tuning on the support set before inference.

### Step A: LoRA Training

```bash
# Standard
bash bash/train.sh

# With frame timestamps (adds [Frame at X.Xs] to temporal questions only)
bash bash/train.sh --frame-timestamps
```

With `--frame-timestamps`, `prepare_train_data.py` runs automatically and replaces `<image>` tags with `[Frame at X.Xs]<image>` for temporal questions before training.

Output: `output/egocross_lora_YYYYMMDD_HHMMSS/` (new directory per run)

Key settings in `configs/lora.yaml`:

| Parameter                     | Default                     | Description                         |
| ----------------------------- | --------------------------- | ----------------------------------- |
| `model_name_or_path`          | `Qwen/Qwen3-VL-4B-Instruct` | Base model                          |
| `lora_rank`                   | `64`                        | LoRA rank                           |
| `num_train_epochs`            | `10`                        | Number of epochs                    |
| `learning_rate`               | `1.0e-5`                    | Learning rate                       |
| `per_device_train_batch_size` | `1`                         | Batch size per GPU                  |
| `gradient_accumulation_steps` | `1`                         | Gradient accumulation steps         |
| `save_strategy`               | `epoch`                     | Save a checkpoint after every epoch |

### Step B: Compare Checkpoints

Adapters can be loaded directly — no merge required.

```bash
# Auto-detect latest LoRA output and evaluate all checkpoints
bash bash/eval_checkpoints.sh

# Specify directory explicitly
bash bash/eval_checkpoints.sh output/egocross_lora_20260510_185322
```

Output: `outputs/eval_checkpoints_XXXXXX_YYYYMMDD.txt` (with per-checkpoint Overall accuracy summary)

To evaluate a single checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 python infer_all.py --mode eval \
    --adapter-path ./output/egocross_lora_20260510_185322/checkpoint-40
```

### Step C: LoRA Merge → Inference (Optional)

Merging is only needed when using the merged model path via `--model`. Skip this step if using `--adapter-path`.

```bash
# Merge LoRA into base model → output to models/egocross_XXXXXX/
.venv/bin/llamafactory-cli export configs/merge_lora.yaml

# Inference with merged model
CUDA_VISIBLE_DEVICES=0 python infer_all.py --mode eval --model egocross_XXXXXX
```

---

## Step 1: Question Type Classification

Classify each support set question by question type and save to `outputs/support_question_types.json`.
Required before running warmup. Can be skipped if the file already exists.

```bash
source .env && python classify_support.py
```

---

## Step 2: Warmup Conversation Generation

### Generate with Gemini

```bash
# AI Studio (API key)
source .env && python warmup_gemini.py

# Vertex AI
source .env && python warmup_gemini.py --use-vertex

# Specify model
source .env && python warmup_gemini.py --model gemini-3.1-flash-image-preview

# Change output path
source .env && python warmup_gemini.py --output outputs/my_warmup.json
```

Output: `outputs/warmup_conversations_gemini.json`

Key options:

| Option               | Default                                    | Description                               |
| -------------------- | ------------------------------------------ | ----------------------------------------- |
| `--model`            | `gemini-3.1-flash-image-preview`           | Model to use                              |
| `--thinking-budget`  | `0`                                        | Thinking token budget (0=off, -1=dynamic) |
| `--rate-limit-sleep` | `1.0`                                      | Sleep seconds between requests            |
| `--use-vertex`       | off                                        | Use Vertex AI instead of AI Studio        |
| `--output`           | `outputs/warmup_conversations_gemini.json` | Output file path                          |

### Generate with Qwen (HuggingFace Hub / LoRA adapter)

```bash
# HuggingFace Hub model ID
CUDA_VISIBLE_DEVICES=0 python warmup_qwen.py --model-id Qwen/Qwen3-VL-4B-Instruct

# LoRA adapter (no merge required)
CUDA_VISIBLE_DEVICES=0 python warmup_qwen.py \
    --adapter-path output/egocross_lora_YYYYMMDD_HHMMSS/checkpoint-480 \
    --output outputs/warmup_ckpt480.json

# LoRA + frame timestamps (use when training used --frame-timestamps)
CUDA_VISIBLE_DEVICES=0 python warmup_qwen.py \
    --adapter-path output/egocross_lora_YYYYMMDD_HHMMSS/checkpoint-480 \
    --frame-timestamps \
    --output outputs/warmup_ckpt480_frame_ts.json

# Base model
CUDA_VISIBLE_DEVICES=0 python warmup_qwen.py --baseline

# Enable thinking mode
CUDA_VISIBLE_DEVICES=0 python warmup_qwen.py --model-id Qwen/Qwen3-VL-4B-Instruct --thinking
```

Output: `outputs/warmup_conversations_qwen.json`

With `--frame-timestamps`, a `"timestamps"` field is saved in each warmup turn. These are automatically injected as `[Frame at X.Xs]` when loading — no flag needed on the inference side for warmup turns (but `--frame-timestamps` is still required for the main question).

Key options:

| Option               | Default                                  | Description                                          |
| -------------------- | ---------------------------------------- | ---------------------------------------------------- |
| `--model`            | —                                        | Directory name under `models/`                       |
| `--baseline`         | off                                      | Use `Qwen/Qwen3-VL-4B-Instruct`                      |
| `--model-id`         | —                                        | HuggingFace Hub model ID (highest priority)          |
| `--adapter-path`     | —                                        | Path to LoRA adapter (no merge required)             |
| `--frame-timestamps` | off                                      | Add `[Frame at X.Xs]` to temporal questions and save |
| `--max-pixels`       | `128000`                                 | Max pixels per frame                                 |
| `--thinking`         | off                                      | Enable thinking mode                                 |
| `--output`           | `outputs/warmup_conversations_qwen.json` | Output file path                                     |

---

## Step 3: Inference

### Gemini Inference

```bash
# eval (check accuracy on support set)
source .env && python infer_gemini.py --mode eval

# test (generate submission predictions)
source .env && python infer_gemini.py --mode test

# Use Vertex AI
source .env && python infer_gemini.py --mode test --use-vertex

# With warmup
source .env && python infer_gemini.py --mode eval \
    --warmup-file outputs/warmup_conversations_gemini.json

# With warmup + frame limit per turn
source .env && python infer_gemini.py --mode eval \
    --warmup-file outputs/warmup_conversations_gemini.json \
    --warmup-max-frames 5

# Visual few-shot (pass support set images+questions+answers directly, no reflection)
source .env && python infer_gemini.py --mode eval --visual-fewshot

# Visual few-shot + frame limit
source .env && python infer_gemini.py --mode eval \
    --visual-fewshot --visual-fewshot-max-frames 5

# Video input mode
source .env && python infer_gemini.py --mode eval --input-mode video

# Limit items / filter domain / resume from ID
source .env && python infer_gemini.py --mode eval --limit 50
source .env && python infer_gemini.py --mode eval --domain surgery
source .env && python infer_gemini.py --mode eval --resume-from-id 42
```

Key options:

| Option                        | Default                          | Description                                                  |
| ----------------------------- | -------------------------------- | ------------------------------------------------------------ |
| `--mode`                      | `eval`                           | `eval`: accuracy check / `test`: generate submission         |
| `--model`                     | `gemini-3.1-flash-image-preview` | Model to use                                                 |
| `--prompt-style`              | `domain`                         | `default` / `domain` (domain-specific system prompt)         |
| `--thinking-budget`           | `0`                              | Thinking token budget (0=off, -1=dynamic)                    |
| `--input-mode`                | `image`                          | `image` / `video`                                            |
| `--fewshot`                   | off                              | Prepend same-question few-shot examples                      |
| `--rate-limit-sleep`          | `0.5`                            | Sleep seconds between requests                               |
| `--use-vertex`                | off                              | Use Vertex AI                                                |
| `--warmup-file`               | —                                | Path to warmup conversation JSON                             |
| `--warmup-max-frames`         | `0`                              | Max frames per warmup turn (0=unlimited)                     |
| `--visual-fewshot`            | off                              | Pass support set images+questions+answers as visual few-shot |
| `--visual-fewshot-max-frames` | `0`                              | Max frames per visual few-shot example (0=unlimited)         |
| `--limit`                     | `0`                              | Process only the first N items                               |
| `--domain`                    | —                                | Process a specific domain only                               |
| `--resume-from-id`            | —                                | Resume processing from the specified item ID                 |

Output: `outputs/predictions_gemini_YYYYMMDD_HHMMSS.json`

### Qwen Local Model Inference

```bash
# eval
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 \
    python infer_all.py --mode eval --model-id Qwen/Qwen3-VL-4B-Instruct

# test (generate submission predictions)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 \
    python infer_all.py --mode test --model-id Qwen/Qwen3-VL-4B-Instruct

# With warmup
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 \
    python infer_all.py --mode eval \
    --model-id Qwen/Qwen3-VL-4B-Instruct \
    --warmup-file outputs/warmup_conversations_qwen.json

# Visual few-shot
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 \
    python infer_all.py --mode eval \
    --model-id Qwen/Qwen3-VL-4B-Instruct \
    --visual-fewshot --visual-fewshot-max-frames 5

# Enable thinking + video input
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 \
    python infer_all.py --mode eval \
    --model-id Qwen/Qwen3-VL-4B-Instruct \
    --thinking --input-mode video

# Thinking model — --thinking is required
# (without it, max_new_tokens=32 cuts off mid-thought and all answers fall back to "A")
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 \
    python infer_all.py --mode test \
    --model-id Qwen/Qwen3-VL-4B-Thinking \
    --warmup-file outputs/warmup_conversations_qwen.json \
    --thinking
```

Key options:

| Option                        | Default   | Description                                                  |
| ----------------------------- | --------- | ------------------------------------------------------------ |
| `--mode`                      | `test`    | `eval`: accuracy check / `test`: generate submission         |
| `--model`                     | —         | Directory name under `models/`                               |
| `--baseline`                  | off       | Use `Qwen/Qwen3-VL-4B-Instruct`                              |
| `--model-id`                  | —         | HuggingFace Hub model ID (highest priority)                  |
| `--adapter-path`              | —         | Path to LoRA adapter (no merge required)                     |
| `--max-pixels`                | `128000`  | Max pixels per frame                                         |
| `--input-mode`                | `image`   | `image` / `video`                                            |
| `--thinking`                  | off       | Enable thinking mode (required for Thinking models)          |
| `--prompt-style`              | `default` | `default` / `clean` / `domain`                               |
| `--fewshot`                   | off       | Prepend same-question few-shot examples                      |
| `--single-model`              | —         | Use one model for all domains                                |
| `--warmup-file`               | —         | Path to warmup conversation JSON                             |
| `--warmup-max-frames`         | `0`       | Max frames per warmup turn (0=unlimited)                     |
| `--visual-fewshot`            | off       | Pass support set images+questions+answers as visual few-shot |
| `--visual-fewshot-max-frames` | `0`       | Max frames per visual few-shot example (0=unlimited)         |
| `--frame-timestamps`          | off       | Add `[Frame at X.Xs]` to temporal question frames            |

Output: `outputs/predictions_YYYYMMDD_HHMMSS.json`

---

## Typical Flow

```bash
# 1. Classify question types
source .env && python classify_support.py

# 2a. Generate warmup with Gemini
source .env && python warmup_gemini.py

# 2b. Generate warmup with Qwen
CUDA_VISIBLE_DEVICES=0 python warmup_qwen.py --model-id Qwen/Qwen3-VL-4B-Instruct

# 3a. Gemini inference (eval)
source .env && python infer_gemini.py --mode eval \
    --warmup-file outputs/warmup_conversations_gemini.json

# 3b. Qwen inference (eval)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 \
    python infer_all.py --mode eval \
    --model-id Qwen/Qwen3-VL-4B-Instruct \
    --warmup-file outputs/warmup_conversations_qwen.json

# 4. Generate test predictions and submit
source .env && python infer_gemini.py --mode test \
    --warmup-file outputs/warmup_conversations_gemini.json
```

---

## Full Pipeline with Frame Timestamps

Use this flow to keep timestamps consistent across training, warmup, and inference.

```bash
# Step 1: Train with frame timestamps
bash bash/train.sh --frame-timestamps
# → checkpoints saved to output/egocross_lora_YYYYMMDD_HHMMSS/

# Step 2: Generate warmup (e.g. epoch 6 = checkpoint-480)
CUDA_VISIBLE_DEVICES=0 python warmup_qwen.py \
    --adapter-path output/egocross_lora_YYYYMMDD_HHMMSS/checkpoint-480 \
    --frame-timestamps \
    --output outputs/warmup_ckpt480_frame_ts.json

# Step 3: Test inference
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 \
    python infer_all.py --mode test \
    --adapter-path output/egocross_lora_YYYYMMDD_HHMMSS/checkpoint-480 \
    --warmup-file outputs/warmup_ckpt480_frame_ts.json \
    --frame-timestamps
```

> **Note**: Keep `--frame-timestamps` consistent across training, warmup, and inference. Timestamps are stored inside the warmup JSON and injected automatically when loading, so the flag is not needed on the inference side for warmup turns — but it is still required for the main question.

---

## outputs/ Directory

| File                                      | Generated by          | Contents                            |
| ----------------------------------------- | --------------------- | ----------------------------------- |
| `support_question_types.json`             | `classify_support.py` | Question type classification        |
| `warmup_conversations_gemini.json`        | `warmup_gemini.py`    | Gemini warmup conversations         |
| `warmup_conversations_qwen.json`          | `warmup_qwen.py`      | Qwen warmup conversations           |
| `predictions_YYYYMMDD_HHMMSS.json`        | `infer_all.py`        | Qwen predictions (for submission)   |
| `predictions_gemini_YYYYMMDD_HHMMSS.json` | `infer_gemini.py`     | Gemini predictions (for submission) |
| `log_*.txt`                               | inference scripts     | Inference logs                      |
