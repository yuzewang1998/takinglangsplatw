#!/bin/bash
CASE_NAME="brandenburg_gate_wegs"

# path to lerf_ovs/label
gt_folder="/media/wangyz/DATA/UBUNTU_data/dataset/PT/label"
dataset_folder="/media/wangyz/DATA/UBUNTU_data/dataset/PT/brandenburg_gate_tiny"
root_path="../"

python evaluate_iou_loc_pt.py \
        --dataset_name ${CASE_NAME} \
        --feat_dir ${dataset_folder} \
        --ae_ckpt_dir ${root_path}/autoencoder/ckpt \
        --output_dir ${root_path}/eval_result \
        --mask_thresh 0.4 \
        --encoder_dims 256 128 64 32 3 \
        --decoder_dims 16 32 64 128 256 256 512 \
        --json_folder ${gt_folder}
