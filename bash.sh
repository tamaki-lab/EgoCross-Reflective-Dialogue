# source .env && python infer_gemini.py --mode test --model gemini-3-pro-image-preview --use-vertex
# # source .env && python infer_gemini.py --mode test --warmup-file outputs/warmup_conversations.json --use-vertex --warmup-max-frames 5
# python infer_gemini.py --use-vertex --project dazzling-pillar-495213-c7
# # Step 1: 質問タイプ分類（既に outputs/support_question_types.json があるならスキップ可）
# source .env && python classify_support.py
# # Step 2: warmup会話を生成
# source .env && python warmup_gemini.py --use-vertex

# TIMESTAMP=$(date +%Y%m%d_%H%M%S)
# WARMUP_FILE="outputs/warmup_gemini_pro_${TIMESTAMP}.json"

# # warmup生成
# python warmup_gemini.py \
#     --model gemini-3.1-pro-preview \
#     --use-vertex \
#     --project dazzling-pillar-495213-c7 \
#     --output "$WARMUP_FILE"

# # test推論
# python infer_gemini.py --mode test \
#     --model gemini-3.1-pro-preview \
#     --use-vertex \
#     --project dazzling-pillar-495213-c7 \
#     --warmup-file "$WARMUP_FILE" \
#     --warmup-max-frames 5


# Step 1: warmup生成（Gemini + フレームタイムスタンプ）
python3 warmup_gemini.py \
    --frame-timestamps \
    --output outputs/warmup_gemini_flash_frame_ts.json \
    --use-vertex \
    --project dazzling-pillar-495213-c7

# Step 2: 推論（test）
python3 infer_gemini.py \
    --frame-timestamps \
    --mode test \
    --warmup-file outputs/warmup_gemini_flash_frame_ts.json \
    --use-vertex \
    --project dazzling-pillar-495213-c7 \
    --warmup-max-frames 8
