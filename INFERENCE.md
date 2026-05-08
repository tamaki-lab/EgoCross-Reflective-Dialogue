# EgoCross 推論・Warmup 実行ガイド

## ファイル構成

| スクリプト            | 役割                                               |
| --------------------- | -------------------------------------------------- |
| `classify_support.py` | support set の問題を question_type に分類 (Step 1) |
| `warmup_gemini.py`    | Gemini で warmup 会話を生成 (Step 2a)              |
| `warmup_qwen.py`      | Qwen ローカルモデルで warmup 会話を生成 (Step 2b)  |
| `infer_gemini.py`     | Gemini で推論                                      |
| `infer_all.py`        | Qwen ローカルモデルで推論                          |

---

## 事前準備

`.env` に API キーを設定:

```
GEMINI_API_KEY=...
GOOGLE_CLOUD_PROJECT=...
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

### Qwen で生成 (HuggingFace Hub 指定)

```bash
# HuggingFace Hub モデルID で指定
CUDA_VISIBLE_DEVICES=0 python warmup_qwen.py --model-id Qwen/Qwen3-VL-4B-Instruct

# ベースモデル使用
CUDA_VISIBLE_DEVICES=0 python warmup_qwen.py --baseline

# thinking モード有効
CUDA_VISIBLE_DEVICES=0 python warmup_qwen.py --model-id Qwen/Qwen3-VL-4B-Instruct --thinking

# 出力先変更
CUDA_VISIBLE_DEVICES=0 python warmup_qwen.py --model-id Qwen/Qwen3-VL-4B-Instruct --output outputs/my_warmup.json
```

出力: `outputs/warmup_conversations_qwen.json`

主なオプション:

| オプション     | デフォルト                               | 説明                                  |
| -------------- | ---------------------------------------- | ------------------------------------- |
| `--model`      | —                                        | `models/` 以下のディレクトリ名        |
| `--baseline`   | off                                      | `Qwen/Qwen3-VL-4B-Instruct` を使用    |
| `--model-id`   | —                                        | HuggingFace Hub モデルID (優先度最高) |
| `--max-pixels` | `128000`                                 | 1フレームあたりの最大ピクセル数       |
| `--thinking`   | off                                      | thinking モードを有効化               |
| `--output`     | `outputs/warmup_conversations_qwen.json` | 出力ファイルパス                      |

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
| `--thinking`                  | off        | thinking モードを有効化                                           |
| `--prompt-style`              | `default`  | `default` / `clean` / `domain`                                    |
| `--fewshot`                   | off        | same-question few-shot を付加                                     |
| `--single-model`              | —          | 全ドメインで同一モデルを使用                                      |
| `--warmup-file`               | —          | warmup 会話 JSON のパス                                           |
| `--warmup-max-frames`         | `0`        | warmup 各ターンのフレーム上限                                     |
| `--visual-fewshot`            | off        | support set の画像+問題+回答をそのまま visual few-shot として渡す |
| `--visual-fewshot-max-frames` | `0`        | visual few-shot 各例のフレーム上限                                |

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

## outputs/ ディレクトリ

| ファイル                                  | 生成元                | 内容                     |
| ----------------------------------------- | --------------------- | ------------------------ |
| `support_question_types.json`             | `classify_support.py` | 問題タイプ分類結果       |
| `warmup_conversations_gemini.json`        | `warmup_gemini.py`    | Gemini 生成 warmup 会話  |
| `warmup_conversations_qwen.json`          | `warmup_qwen.py`      | Qwen 生成 warmup 会話    |
| `predictions_YYYYMMDD_HHMMSS.json`        | `infer_all.py`        | Qwen 推論結果 (提出用)   |
| `predictions_gemini_YYYYMMDD_HHMMSS.json` | `infer_gemini.py`     | Gemini 推論結果 (提出用) |
| `log_*.txt`                               | 各推論スクリプト      | 推論ログ                 |
