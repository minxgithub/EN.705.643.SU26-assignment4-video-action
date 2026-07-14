#!/bin/bash
FRAME_DIR="${FRAME_DIR:-/content/datasets/HMDB51}"
CNN_BACKBONE="${CNN_BACKBONE:-resnet50}"
CKPT_PATH="${CKPT_PATH:-./models/best_model_wts.pt}"
WANDB_PROJECT="${WANDB_PROJECT:-assignment-4-training}"
RUN_NAME="${RUN_NAME:-lrcn-resnet50-uniform-sampling-valid-lengths}"

python run.py \
    --frame_dir "$FRAME_DIR" \
    --ckpt "$CKPT_PATH" \
    --model_type lrcn \
    --cnn_backbone "$CNN_BACKBONE" \
    --n_classes 51 \
    --fr_per_vid 16 \
    --batch_size 4 \
    --wandb_project "$WANDB_PROJECT" \
    --run_name "$RUN_NAME" \
    --mode eval
