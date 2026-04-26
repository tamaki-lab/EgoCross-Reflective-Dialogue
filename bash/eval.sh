PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=1 python3 infer_all.py --mode eval --max-pixels 128000


# 画像モード（現在のベースライン）
# python3 infer_all.py --mode eval --input-mode image --max-pixels 128000

# 動画モード
# python3 infer_all.py --mode eval --input-mode video --max-pixels 128000
