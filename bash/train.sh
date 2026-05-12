#!/bin/bash
# Usage: bash bash/train.sh [--frame-timestamps]
# --frame-timestamps: temporal問題に [Frame at X.Xs] を追加したデータで学習

FRAME_TS=0
for arg in "$@"; do
    if [ "$arg" = "--frame-timestamps" ]; then
        FRAME_TS=1
    fi
done

OUTPUT_DIR="./output/egocross_lora_$(date +%Y%m%d_%H%M%S)"
echo "Output dir: $OUTPUT_DIR"

if [ "$FRAME_TS" = "1" ]; then
    echo "Frame timestamps mode: running prepare_train_data.py..."
    python3 prepare_train_data.py
    DATASET="egocross_frame_ts"
else
    DATASET="egocross"
fi

TEMP_CONFIG=$(mktemp /tmp/lora_XXXXXX.yaml)
sed "s|output_dir:.*|output_dir: $OUTPUT_DIR|;s|dataset:.*|dataset: $DATASET|" configs/lora.yaml > "$TEMP_CONFIG"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 \
    .venv/bin/llamafactory-cli train "$TEMP_CONFIG"

rm "$TEMP_CONFIG"
