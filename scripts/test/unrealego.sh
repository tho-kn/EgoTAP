#!/bin/bash

# script
python test.py \
    --project_name UnrealEgoPose \
    --experiment_name egotap_unrealego \
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
    --batch_size 32 \
    --num_rot_heatmap 15 \
    --num_heatmap 15 \
    --heatmap_type sin \
\
