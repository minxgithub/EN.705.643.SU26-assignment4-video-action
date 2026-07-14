#!/bin/bash
FRAME_DIR="${FRAME_DIR:-HMDB51}"
N_EPOCHS="${N_EPOCHS:-50}"

python run.py \
    --frame_dir "$FRAME_DIR" \
    --train_size 0.75 \
    --test_size 0.15 \
    --fr_per_vid 16 \
    --n_classes 51 \
    --model_type lrcn \
    --cnn_backbone resnet50 \
    --wandb_project assignment-4-training \
    --run_name lrcn-resnet50-baseline \
    --batch_size 4 \
    --n_epochs "$N_EPOCHS" \
    --mode train
