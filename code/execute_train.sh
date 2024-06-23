python3 ${T5X_DIR}/t5x/train.py \
  --gin_search_paths=${PROJECT_DIR} \
  --gin_file="model.gin" \
  --gin_file="finetune.gin" \
  --gin.MODEL_DIR=\"TFM/code/\" \
  --gin.CHECKPOINT_PATH=\"gs://tfm-jazz-transcription-marvin/checkpoints/checkpoint_1\"