OUTPUT_DIR="./output/egocross_lora_$(date +%Y%m%d_%H%M%S)"
echo "Output dir: $OUTPUT_DIR"

TEMP_CONFIG=$(mktemp /tmp/lora_XXXXXX.yaml)
sed "s|output_dir:.*|output_dir: $OUTPUT_DIR|" configs/lora.yaml > "$TEMP_CONFIG"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 \
    .venv/bin/llamafactory-cli train "$TEMP_CONFIG"

rm "$TEMP_CONFIG"
