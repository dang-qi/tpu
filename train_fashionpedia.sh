MODEL_DIR="~/Vision/git/tpu/output_models"
TRAIN_FILE_PATTERN=~/data/datasets/Fashionpedia/tf_record/train-*
EVAL_FILE_PATTERN=~/data/datasets/Fashionpedia/tf_record/val-*
VAL_JSON_FILE=~/data/datasets/Fashionpedia/annotations/instances_attributes_val2020.json
RESNET_CHECKPOINT="gs://cloud-tpu-artifacts/resnet/resnet-nhwc-2018-10-14/model.ckpt-112602"
RESNET_CHECKPOINT="gs://cloud-tpu-checkpoints/retinanet/resnet50-checkpoint-2018-02-07"
CONFIG_FILE_PATH=~/Vision/git/tpu/models/official/mask_rcnn/configs/cloud/gpu-4.yaml
python ~/Vision/git/tpu/models/official/detection/main.py \
  --model="mask_rcnn" \
  --model_dir="${MODEL_DIR?}" \
  --mode=train \
  --eval_after_training=True \
  --use_tpu=False \
  --params_override="{ train: { checkpoint: { path: ${RESNET_CHECKPOINT?}, prefix: resnet50/ }, train_file_pattern: ${TRAIN_FILE_PATTERN?} }, eval: { val_json_file: ${VAL_JSON_FILE?}, eval_file_pattern: ${EVAL_FILE_PATTERN?} } }"
