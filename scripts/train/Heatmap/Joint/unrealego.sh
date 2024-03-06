#!/bin/bash

# script
python train.py \
    --project_name UnrealEgoPose \
    --experiment_name unrealego_heatmap_shared_pos \
    --model heatmap_shared \
\
\
    --use_amp \
    --init_ImageNet \
    --auto_restart \
    --optimizer_type Adam \
    --lr 1e-3 \
\
    --lambda_mpjpe 0.1 \
    --lambda_heatmap 1.0 \
    --lambda_rot_heatmap 1.0 \
    --lambda_cos_sim -0.01 \
    --lambda_heatmap_rec 0.001 \
    --lambda_rot_heatmap_rec 0.001 \
    --gpu_ids 3 \
\
    --niter 5 \
    --niter_decay 5 \
    --batch_size 16 \
    --num_rot_heatmap 0 \
    --num_heatmap 15 \
    --data_dir /ssd_data1/UnrealEgoData/ \
\
