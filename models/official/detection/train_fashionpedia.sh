MODEL_DIR="~/Vision/git/tpu/output_models"
TRAIN_FILE_PATTERN="~/data/datasets/Fashionpedia/tf_record/train"
EVAL_FILE_PATTERN="~/data/datasets/Fashionpedia/tf_record/val"
VAL_JSON_FILE="~/data/datasets/Fashionpedia/annotations/instances_attributes_val2020.json"
RESNET_CHECKPOINT="gs://cloud-tpu-artifacts/resnet/resnet-nhwc-2018-10-14/model.ckpt-112602"
python main.py \
  --model="mask_rcnn" \
  --model_dir="${MODEL_DIR?}" \
  --mode=train \
  --eval_after_training=True \
  --use_tpu=False \
  --params_override="{ train: { checkpoint: { path: ${RESNET_CHECKPOINT?}, prefix: resnet50/ }, train_file_pattern: ${TRAIN_FILE_PATTERN?} }, eval: { val_json_file: ${VAL_JSON_FILE?}, eval_file_pattern: ${EVAL_FILE_PATTERN?} } }"
