#!/bin/bash
python run.py \
    --ckpt ./models/best_model_wts.pt \
    --model_type lrcn \
    --n_classes 51 \
    --fr_per_vid 16 \
    --batch_size 4 \
    --mode eval
