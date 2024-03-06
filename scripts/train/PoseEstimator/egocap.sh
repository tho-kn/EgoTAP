#!/bin/bash

# script
python train.py \
    --project_name UnrealEgoPose \
    --experiment_name egotap_egocap \
    --model egotap_autoencoder \
\
\
    --use_amp \
    --init_ImageNet \
    --optimizer_type AdamW \
    --lr_policy cos_anneal_warmup \
    --lr 1e-3 \
\
    --gpu_ids 1 \
    --lambda_mpjpe 0.1 \
    --lambda_rot 1.0 \
    --lambda_indep_pos 0.1 \
    --lambda_heatmap 1.0 \
    --lambda_rot_heatmap 1.0 \
    --lambda_cos_sim -0.01 \
    --lambda_heatmap_rec 0.0 \
    --lambda_rot_heatmap_rec 0.0 \
    --skel_layer PU \
    --ae_hidden_size 128 \
    --patched_heatmap_ae \
\
    --epoch_count 1 \
    --niter 2 \
    --niter_decay 15 \
    --batch_size 32 \
    --num_rot_heatmap 17 \
    --num_heatmap 17 \
    --heatmap_type sin \
    --data_dir /data/EgoCap/ \
    --joint_preset EgoCap \
    --path_to_trained_heatmap ./log/egocap_heatmap_shared/best_net_HeatMap.pth \
