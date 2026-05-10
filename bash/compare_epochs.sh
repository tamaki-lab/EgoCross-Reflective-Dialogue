#!/bin/bash
# fine-tuning あり/なし × warmup あり/なし の比較
# 指定epochのチェックポイントについて:
#   (A) plain test/eval
#   (B) warmup生成 → warmup付き test/eval
# を実行してサマリーを出力する
#
# 使い方: bash bash/compare_epochs.sh [LoRAディレクトリ] [epoch1] [epoch2] ...
# 例: bash bash/compare_epochs.sh output/egocross_lora_20260510_202726 3 6 9
#     bash bash/compare_epochs.sh  # 最新ディレクトリ + epoch 3 6 9

LORA_DIR="${1:-$(ls -td output/egocross_lora_* 2>/dev/null | head -1)}"
shift 2>/dev/null
EPOCHS="${@:-3 6 9}"

if [ -z "$LORA_DIR" ]; then
    echo "ERROR: LoRAディレクトリが見つかりません"
    exit 1
fi

# stepsPerEpochを自動計算 (最小チェックポイント番号 = 1epoch分のステップ数)
FIRST_CKPT=$(ls -d "$LORA_DIR"/checkpoint-* 2>/dev/null | sort -t'-' -k2 -n | head -1)
STEPS_PER_EPOCH=$(basename "$FIRST_CKPT" | sed 's/checkpoint-//')

echo "対象ディレクトリ : $LORA_DIR"
echo "ステップ/epoch  : $STEPS_PER_EPOCH"
echo "対象epoch       : $EPOCHS"
echo ""

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_FILE="outputs/compare_epochs_$(basename $LORA_DIR)_${TIMESTAMP}.txt"
mkdir -p outputs

EVAL_SUMMARY=()

for EPOCH in $EPOCHS; do
    STEP=$((STEPS_PER_EPOCH * EPOCH))
    CKPT="$LORA_DIR/checkpoint-$STEP"

    if [ ! -d "$CKPT" ]; then
        echo "WARNING: $CKPT が存在しません、スキップします"
        continue
    fi

    echo "========================================================"
    echo "Epoch $EPOCH (checkpoint-$STEP)"
    echo "========================================================"

    # --- (A) warmup なし ---
    echo "[A] plain eval  (epoch $EPOCH, no warmup)"
    EVAL_OUT=$(PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 \
        python infer_all.py --mode eval --adapter-path "$CKPT" 2>&1)
    echo "$EVAL_OUT" | tail -8
    OVERALL_A=$(echo "$EVAL_OUT" | grep "Overall" | tail -1)
    EVAL_SUMMARY+=("epoch${EPOCH} plain  : $OVERALL_A")

    echo ""
    echo "[A] plain test  (epoch $EPOCH, no warmup)"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 \
        python infer_all.py --mode test --adapter-path "$CKPT" 2>&1 | tail -3

    echo ""

    # --- warmup 生成 ---
    WARMUP_FILE="outputs/warmup_epoch${EPOCH}_${TIMESTAMP}.json"
    echo "[warmup生成] epoch $EPOCH → $WARMUP_FILE"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 \
        python warmup_qwen.py --adapter-path "$CKPT" --output "$WARMUP_FILE" 2>&1 | tail -5

    echo ""

    # --- (B) warmup あり ---
    echo "[B] warmup eval (epoch $EPOCH, with warmup)"
    EVAL_OUT=$(PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 \
        python infer_all.py --mode eval --adapter-path "$CKPT" \
        --warmup-file "$WARMUP_FILE" 2>&1)
    echo "$EVAL_OUT" | tail -8
    OVERALL_B=$(echo "$EVAL_OUT" | grep "Overall" | tail -1)
    EVAL_SUMMARY+=("epoch${EPOCH} warmup : $OVERALL_B")

    echo ""
    echo "[B] warmup test (epoch $EPOCH, with warmup)"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 \
        python infer_all.py --mode test --adapter-path "$CKPT" \
        --warmup-file "$WARMUP_FILE" 2>&1 | tail -3

    echo ""
done

# サマリー
echo "========================================================"
echo "=== Eval 比較サマリー ==="
echo "========================================================"
for LINE in "${EVAL_SUMMARY[@]}"; do
    echo "  $LINE"
done

{
    echo "=== Eval 比較サマリー ==="
    echo "対象: $LORA_DIR"
    echo "実行日時: $(date)"
    echo ""
    for LINE in "${EVAL_SUMMARY[@]}"; do
        echo "  $LINE"
    done
} > "$RESULT_FILE"

echo ""
echo "結果を保存しました: $RESULT_FILE"
