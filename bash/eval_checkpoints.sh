#!/bin/bash
# 指定したLoRA出力ディレクトリの全チェックポイントをeval
# 使い方: bash bash/eval_checkpoints.sh [出力ディレクトリ]
# 例: bash bash/eval_checkpoints.sh output/egocross_lora_20260510_185322
#     bash bash/eval_checkpoints.sh  # 最新のoutput/egocross_lora_*を使用

LORA_DIR="${1:-$(ls -td output/egocross_lora_* 2>/dev/null | head -1)}"

if [ -z "$LORA_DIR" ]; then
    echo "ERROR: LoRAディレクトリが見つかりません"
    exit 1
fi

echo "対象ディレクトリ: $LORA_DIR"
RESULT_FILE="outputs/eval_checkpoints_$(basename $LORA_DIR)_$(date +%Y%m%d_%H%M%S).txt"
echo "結果保存先: $RESULT_FILE"
echo ""

# チェックポイントをステップ番号順にソート
CHECKPOINTS=$(ls -d "$LORA_DIR"/checkpoint-* 2>/dev/null | sort -t'-' -k2 -n)

if [ -z "$CHECKPOINTS" ]; then
    echo "ERROR: チェックポイントが見つかりません: $LORA_DIR/checkpoint-*"
    exit 1
fi

echo "チェックポイント一覧:"
echo "$CHECKPOINTS"
echo ""

# 結果サマリー用
SUMMARY_LINES=()

for CKPT in $CHECKPOINTS; do
    CKPT_NAME=$(basename "$CKPT")
    echo "================================================"
    echo "評価中: $CKPT_NAME"
    echo "================================================"

    OUTPUT=$(PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 \
        python infer_all.py --mode eval --adapter-path "$CKPT" 2>&1)
    echo "$OUTPUT"

    # Overall行を抽出
    OVERALL=$(echo "$OUTPUT" | grep "Overall" | tail -1)
    SUMMARY_LINES+=("$CKPT_NAME: $OVERALL")

    echo ""
done

# サマリー表示と保存
echo "================================================"
echo "=== チェックポイント比較サマリー ==="
echo "================================================"
for LINE in "${SUMMARY_LINES[@]}"; do
    echo "$LINE"
done

{
    echo "=== チェックポイント比較サマリー ==="
    echo "対象: $LORA_DIR"
    echo "実行日時: $(date)"
    echo ""
    for LINE in "${SUMMARY_LINES[@]}"; do
        echo "$LINE"
    done
} > "$RESULT_FILE"

echo ""
echo "結果を保存しました: $RESULT_FILE"
