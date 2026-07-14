#!/bin/bash
FRAME_DIR="${FRAME_DIR:-/content/datasets/HMDB51}"
CNN_BACKBONE="${CNN_BACKBONE:-resnet50}"
DROPOUT="${DROPOUT:-0.5}"
N_EPOCHS="${N_EPOCHS:-50}"
WANDB_PROJECT="${WANDB_PROJECT:-assignment-4-training}"
RUN_NAME="${RUN_NAME:-lrcn-resnet50-uniform-sampling-valid-lengths}"

python run.py \
    --frame_dir "$FRAME_DIR" \
    --train_size 0.75 \
    --test_size 0.15 \
    --n_classes 51 \
    --fr_per_vid 16 \
    --batch_size 4 \
    --model_type lrcn \
    --cnn_backbone "$CNN_BACKBONE" \
    --dropout "$DROPOUT" \
    --wandb_project "$WANDB_PROJECT" \
    --run_name "$RUN_NAME" \
    --n_epochs "$N_EPOCHS" \
    --mode train
