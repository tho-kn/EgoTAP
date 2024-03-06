#!/bin/bash

# script
python test.py \
    --project_name UnrealEgoPose \
    --experiment_name egotap_egocap \
    --model egotap_autoencoder \
\
\
    --use_amp \
\
    --gpu_ids 1 \
    --patched_heatmap_ae \
    --skel_layer PU \
    --ae_hidden_size 128 \
\
    --batch_size 16 \
    --num_rot_heatmap 17 \
    --num_heatmap 17 \
    --heatmap_type sin \
    --data_dir /data/EgoCap/ \
    --joint_preset EgoCap \
