# EgoCross 推論・Warmup 実行ガイド

## ファイル構成

| スクリプト                 | 役割                                               |
| -------------------------- | -------------------------------------------------- |
| `classify_support.py`      | support set の問題を question_type に分類 (Step 1) |
| `warmup_gemini.py`         | Gemini で warmup 会話を生成 (Step 2a)              |
| `warmup_qwen.py`           | Qwen ローカルモデルで warmup 会話を生成 (Step 2b)  |
| `infer_gemini.py`          | Gemini で推論                                      |
| `infer_all.py`             | Qwen ローカルモデルで推論                          |
| `bash/train.sh`            | support set で LoRA fine-tuning                    |
| `bash/eval_checkpoints.sh` | 各チェックポイントの正解率を一括比較               |

---

## 事前準備

`.env` に API キーを設定:

```
GEMINI_API_KEY=...
GOOGLE_CLOUD_PROJECT=...
```

---

## LoRA Fine-tuning フロー

support set で fine-tuning してから推論する場合のフロー。

### Step A: LoRA 学習

```bash
# 通常
bash bash/train.sh

# フレームタイムスタンプあり（temporal問題のみ [Frame at X.Xs] を追加）
bash bash/train.sh --frame-timestamps
```

`--frame-timestamps` を指定すると `prepare_train_data.py` が自動実行され、
temporal問題の `<image>` を `[Frame at X.Xs]<image>` に置換したデータで学習する。

出力: `output/egocross_lora_YYYYMMDD_HHMMSS/`（実行ごとに別ディレクトリ）

設定は `configs/lora.yaml` で管理。主な設定:

| 設定                          | デフォルト                  | 説明                               |
| ----------------------------- | --------------------------- | ---------------------------------- |
| `model_name_or_path`          | `Qwen/Qwen3-VL-4B-Instruct` | ベースモデル                       |
| `lora_rank`                   | `64`                        | LoRA ランク                        |
| `num_train_epochs`            | `10`                        | エポック数                         |
| `learning_rate`               | `1.0e-4`                    | 学習率                             |
| `per_device_train_batch_size` | `1`                         | バッチサイズ                       |
| `gradient_accumulation_steps` | `8`                         | 勾配累積ステップ数                 |
| `save_strategy`               | `epoch`                     | エポックごとにチェックポイント保存 |

### Step B: チェックポイント別正解率の比較

マージ不要で各 epoch のモデルを直接評価できる。

```bash
# 最新のLoRA出力を自動検出して全チェックポイントを評価
bash bash/eval_checkpoints.sh

# ディレクトリを明示指定
bash bash/eval_checkpoints.sh output/egocross_lora_20260510_185322
```

出力: `outputs/eval_checkpoints_XXXXXX_YYYYMMDD.txt`（チェックポイントごとの Overall 正解率サマリー付き）

特定のチェックポイントだけ評価したい場合は `--adapter-path` を使う:

```bash
CUDA_VISIBLE_DEVICES=0 python infer_all.py --mode eval \
    --adapter-path ./output/egocross_lora_20260510_185322/checkpoint-40
```

### Step C: LoRA マージ → 推論

最良のチェックポイントが決まったら `configs/merge_lora.yaml` の `adapter_name_or_path` を更新してマージ:

```bash
# LoRAをベースモデルにマージ → models/egocross_XXXXXX/ に出力
.venv/bin/llamafactory-cli export configs/merge_lora.yaml

# マージ済みモデルで推論
CUDA_VISIBLE_DEVICES=0 python infer_all.py --mode eval --model egocross_XXXXXX
```

---

## Step 1: 問題タイプ分類

support set の各問題を question_type に分類し、`outputs/support_question_types.json` を生成する。  
warmup を使う場合は事前に必須。既にファイルがあればスキップ可。

```bash
source .env && python classify_support.py
```

---

## Step 2: Warmup 会話生成

### Gemini で生成

```bash
# AI Studio (API Key)
source .env && python warmup_gemini.py

# Vertex AI
source .env && python warmup_gemini.py --use-vertex

# モデル指定
source .env && python warmup_gemini.py --model gemini-2.5-flash

# 出力先変更
source .env && python warmup_gemini.py --output outputs/my_warmup.json
```

出力: `outputs/warmup_conversations_gemini.json`

主なオプション:

| オプション           | デフォルト                                 | 説明                                    |
| -------------------- | ------------------------------------------ | --------------------------------------- |
| `--model`            | `gemini-3.1-flash-image-preview`           | 使用モデル                              |
| `--thinking-budget`  | `0`                                        | thinking トークン上限 (0=無効, -1=動的) |
| `--rate-limit-sleep` | `1.0`                                      | リクエスト間 sleep 秒数                 |
| `--use-vertex`       | off                                        | Vertex AI を使用                        |
| `--output`           | `outputs/warmup_conversations_gemini.json` | 出力ファイルパス                        |
| `--explain-correct`  | off                                        | 正解時にも根拠説明をモデルに生成させる  |

### Qwen で生成 (HuggingFace Hub 指定 / LoRA アダプタ指定)

```bash
# HuggingFace Hub モデルID で指定
CUDA_VISIBLE_DEVICES=0 python warmup_qwen.py --model-id Qwen/Qwen3-VL-4B-Instruct

# LoRA アダプタ指定（マージ不要）
CUDA_VISIBLE_DEVICES=0 python warmup_qwen.py \
    --adapter-path output/egocross_lora_YYYYMMDD_HHMMSS/checkpoint-480 \
    --output outputs/warmup_ckpt480.json

# LoRA + フレームタイムスタンプあり（学習時と統一する場合）
CUDA_VISIBLE_DEVICES=0 python warmup_qwen.py \
    --adapter-path output/egocross_lora_YYYYMMDD_HHMMSS/checkpoint-480 \
    --frame-timestamps \
    --output outputs/warmup_ckpt480_frame_ts.json

# ベースモデル使用
CUDA_VISIBLE_DEVICES=0 python warmup_qwen.py --baseline

# thinking モード有効
CUDA_VISIBLE_DEVICES=0 python warmup_qwen.py --model-id Qwen/Qwen3-VL-4B-Instruct --thinking
```

出力: `outputs/warmup_conversations_qwen.json`

`--frame-timestamps` を使うと、temporal問題のターンに `"timestamps"` フィールドが保存される。
ロード時に自動で `[Frame at X.Xs]` が差し込まれるため、推論側でフラグ指定は不要。

主なオプション:

| オプション           | デフォルト                               | 説明                                                |
| -------------------- | ---------------------------------------- | --------------------------------------------------- |
| `--model`            | —                                        | `models/` 以下のディレクトリ名                      |
| `--baseline`         | off                                      | `Qwen/Qwen3-VL-4B-Instruct` を使用                  |
| `--model-id`         | —                                        | HuggingFace Hub モデルID (優先度最高)               |
| `--adapter-path`     | —                                        | LoRA アダプタのパス（マージ不要）                   |
| `--frame-timestamps` | off                                      | temporal問題に `[Frame at X.Xs]` を追加してJSON保存 |
| `--max-pixels`       | `128000`                                 | 1フレームあたりの最大ピクセル数                     |
| `--thinking`         | off                                      | thinking モードを有効化                             |
| `--output`           | `outputs/warmup_conversations_qwen.json` | 出力ファイルパス                                    |
| `--explain-correct`  | off                                      | 正解時にも根拠説明をモデルに生成させる              |

---

## Step 3: 推論

### Gemini 推論

```bash
# eval (support set で正解率確認)
source .env && python infer_gemini.py --mode eval

# test (提出用予測を生成)
source .env && python infer_gemini.py --mode test

# Vertex AI 使用
source .env && python infer_gemini.py --mode test --use-vertex

# warmup あり
source .env && python infer_gemini.py --mode eval \
    --warmup-file outputs/warmup_conversations_gemini.json

# warmup + フレーム数制限
source .env && python infer_gemini.py --mode eval \
    --warmup-file outputs/warmup_conversations_gemini.json \
    --warmup-max-frames 5

# visual few-shot (support setの画像+問題+回答をそのまま渡す、reflection/thinkingなし)
source .env && python infer_gemini.py --mode eval --visual-fewshot

# visual few-shot + フレーム数制限
source .env && python infer_gemini.py --mode eval \
    --visual-fewshot --visual-fewshot-max-frames 5

# モデル・プロンプト変更
source .env && python infer_gemini.py --mode eval \
    --model gemini-2.5-pro \
    --prompt-style domain

# thinking 有効
source .env && python infer_gemini.py --mode eval --thinking-budget 8192

# 動画入力モード
source .env && python infer_gemini.py --mode eval --input-mode video

# 件数制限・ドメイン絞り込み・途中再開
source .env && python infer_gemini.py --mode eval --limit 50
source .env && python infer_gemini.py --mode eval --domain surgery
source .env && python infer_gemini.py --mode eval --resume-from-id 42
```

主なオプション:

| オプション                    | デフォルト                       | 説明                                                              |
| ----------------------------- | -------------------------------- | ----------------------------------------------------------------- |
| `--mode`                      | `eval`                           | `eval`: 正解率確認 / `test`: 提出用予測                           |
| `--model`                     | `gemini-3.1-flash-image-preview` | 使用モデル                                                        |
| `--prompt-style`              | `domain`                         | `default` / `domain` (ドメイン別システムプロンプト)               |
| `--thinking-budget`           | `0`                              | thinking トークン上限 (0=無効, -1=動的)                           |
| `--input-mode`                | `image`                          | `image` / `video`                                                 |
| `--fewshot`                   | off                              | same-question few-shot を付加                                     |
| `--rate-limit-sleep`          | `0.5`                            | リクエスト間 sleep 秒数                                           |
| `--use-vertex`                | off                              | Vertex AI を使用                                                  |
| `--warmup-file`               | —                                | warmup 会話 JSON のパス                                           |
| `--warmup-max-frames`         | `0`                              | warmup 各ターンのフレーム上限                                     |
| `--visual-fewshot`            | off                              | support set の画像+問題+回答をそのまま visual few-shot として渡す |
| `--visual-fewshot-max-frames` | `0`                              | visual few-shot 各例のフレーム上限                                |
| `--limit`                     | `0`                              | 先頭 N 件のみ処理                                                 |
| `--domain`                    | —                                | 特定ドメインのみ処理                                              |
| `--resume-from-id`            | —                                | 指定 ID から末尾まで処理                                          |

出力: `outputs/predictions_gemini_YYYYMMDD_HHMMSS.json`

### Qwen ローカルモデル推論

```bash
# eval
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 \
    python infer_all.py --mode eval --model-id Qwen/Qwen3-VL-4B-Instruct

# test (提出用予測を生成)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 \
    python infer_all.py --mode test --model-id Qwen/Qwen3-VL-4B-Instruct

# warmup あり
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 \
    python infer_all.py --mode eval \
    --model-id Qwen/Qwen3-VL-4B-Instruct \
    --warmup-file outputs/warmup_conversations_qwen.json

# visual few-shot (support setの画像+問題+回答をそのまま渡す、reflection/thinkingなし)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 \
    python infer_all.py --mode eval \
    --model-id Qwen/Qwen3-VL-4B-Instruct \
    --visual-fewshot --visual-fewshot-max-frames 5

# thinking 有効 + 動画入力
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 \
    python infer_all.py --mode eval \
    --model-id Qwen/Qwen3-VL-4B-Instruct \
    --thinking --input-mode video

# Thinking モデル (Qwen3-VL-*-Thinking) を使う場合は --thinking 必須
# (付けないと max_new_tokens=32 になり思考途中で打ち切られ全問 A にフォールバックする)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 \
    python infer_all.py --mode test \
    --model-id Qwen/Qwen3-VL-4B-Thinking \
    --warmup-file outputs/warmup_conversations_qwen.json \
    --thinking

# ドメイン別プロンプト + few-shot
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 \
    python infer_all.py --mode eval \
    --model-id Qwen/Qwen3-VL-4B-Instruct \
    --prompt-style domain --fewshot
```

主なオプション:

| オプション                    | デフォルト | 説明                                                              |
| ----------------------------- | ---------- | ----------------------------------------------------------------- |
| `--mode`                      | `test`     | `eval`: 正解率確認 / `test`: 提出用予測                           |
| `--model`                     | —          | `models/` 以下のディレクトリ名                                    |
| `--baseline`                  | off        | `Qwen/Qwen3-VL-4B-Instruct` を使用                                |
| `--model-id`                  | —          | HuggingFace Hub モデルID (優先度最高)                             |
| `--max-pixels`                | `128000`   | 1フレームあたりの最大ピクセル数                                   |
| `--input-mode`                | `image`    | `image` / `video`                                                 |
| `--thinking`                  | off        | thinking モードを有効化 (Thinking モデル使用時は必須)             |
| `--prompt-style`              | `default`  | `default` / `clean` / `domain`                                    |
| `--fewshot`                   | off        | same-question few-shot を付加                                     |
| `--single-model`              | —          | 全ドメインで同一モデルを使用                                      |
| `--warmup-file`               | —          | warmup 会話 JSON のパス                                           |
| `--warmup-max-frames`         | `0`        | warmup 各ターンのフレーム上限                                     |
| `--visual-fewshot`            | off        | support set の画像+問題+回答をそのまま visual few-shot として渡す |
| `--visual-fewshot-max-frames` | `0`        | visual few-shot 各例のフレーム上限                                |
| `--adapter-path`              | —          | LoRA アダプタのパスを直接指定（マージ不要）                       |
| `--frame-timestamps`          | off        | temporal問題のフレームに `[Frame at X.Xs]` を追加                 |

出力: `outputs/predictions_YYYYMMDD_HHMMSS.json`

---

## 典型的なフロー

```bash
# 1. 問題分類
source .env && python classify_support.py

# 2a. Gemini で warmup 生成
source .env && python warmup_gemini.py

# 2b. Qwen で warmup 生成
CUDA_VISIBLE_DEVICES=0 python warmup_qwen.py --model-id Qwen/Qwen3-VL-4B-Instruct

# 3a. Gemini で推論 (eval)
source .env && python infer_gemini.py --mode eval \
    --warmup-file outputs/warmup_conversations_gemini.json

# 3b. Qwen で推論 (eval)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 \
    python infer_all.py --mode eval \
    --model-id Qwen/Qwen3-VL-4B-Instruct \
    --warmup-file outputs/warmup_conversations_qwen.json

# 4. テスト用予測を生成して提出
source .env && python infer_gemini.py --mode test \
    --warmup-file outputs/warmup_conversations_gemini.json
```

---

## LoRA + フレームタイムスタンプ フルパイプライン

学習・warmup・推論すべてでタイムスタンプを一致させる場合のフロー。

```bash
# Step 1: 学習（temporal問題にタイムスタンプ付き）
bash bash/train.sh --frame-timestamps
# → output/egocross_lora_YYYYMMDD_HHMMSS/ に checkpoint-80〜800 が保存される

# Step 2: warmup 生成（6epoch = checkpoint-480）
CUDA_VISIBLE_DEVICES=0 python warmup_qwen.py \
    --adapter-path output/egocross_lora_YYYYMMDD_HHMMSS/checkpoint-480 \
    --frame-timestamps \
    --output outputs/warmup_ckpt480_frame_ts.json

# Step 3: test 推論
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 \
    python infer_all.py --mode test \
    --adapter-path output/egocross_lora_YYYYMMDD_HHMMSS/checkpoint-480 \
    --warmup-file outputs/warmup_ckpt480_frame_ts.json \
    --frame-timestamps
```

> **注意**: `--frame-timestamps` は学習・warmup・推論で必ず統一すること。
> warmup の JSON にタイムスタンプが保存されているため、
> ロード時に自動で差し込まれる（warmupは推論側でフラグ不要だが、本問には必要）。

---

## outputs/ ディレクトリ

| ファイル                                  | 生成元                | 内容                     |
| ----------------------------------------- | --------------------- | ------------------------ |
| `support_question_types.json`             | `classify_support.py` | 問題タイプ分類結果       |
| `warmup_conversations_gemini.json`        | `warmup_gemini.py`    | Gemini 生成 warmup 会話  |
| `warmup_conversations_qwen.json`          | `warmup_qwen.py`      | Qwen 生成 warmup 会話    |
| `predictions_YYYYMMDD_HHMMSS.json`        | `infer_all.py`        | Qwen 推論結果 (提出用)   |
| `predictions_gemini_YYYYMMDD_HHMMSS.json` | `infer_gemini.py`     | Gemini 推論結果 (提出用) |
| `log_*.txt`                               | 各推論スクリプト      | 推論ログ                 |
