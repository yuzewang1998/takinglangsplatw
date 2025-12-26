source ~/anaconda3/etc/profile.d/conda.sh
conda activate malegs

ulimit -n 10240

# If dont change dataset (Brandburg,Trevi Fountain),don't change these settings.
scene_dir=/media/wangyz/DATA/UBUNTU_data/dataset/PT/brandenburg_gate_tiny
# If dont change the 3D Reconstruction architecture (3DGS OR WEGS), don't change these settings.
ReconCodebase='wegs'
ckpt_iter='200000'
reconstruction_case_name=brandenburg_gate_${ReconCodebase} # # save the trained WE-GS or vanilla 3DGS into the dataset folder
#
# go ahead
FeatureExtractMethod='VanillaCLIPExtUncetainDM-T-100epoch'
LangReconMethod='catFeature12dReconUncertainlyTMAM60Kiter_brandenburg_gate'
exp_name=${ReconCodebase}_${FeatureExtractMethod}_${LangReconMethod} # wegs_SimpleAugOrigin3var-300epoch_VanillaRecon
CASE_NAME=${exp_name} # 可以不动，只是标注
which_feature_fusion_func='aug_wUncertainly_TMAM' # default: use vanilla LangSplat; aug: with 3 aug-view; aug_wUncertainly_TM; aug_wUncertainly:WITH UNCERTAINLY MAP. This parameter only used in the Language Reconstruction Pipeline. For eval, use another: which_post_feature_fusion_func.
# train&test AE
cd autoencoder
python train.py --dataset_path ${scene_dir} --dataset_name ${CASE_NAME} --train_feature_func default --num_epochs 100 --train_with_uncertainly_map --fusion_uncertainly_map_func direct_multiply
python test.py --dataset_path ${scene_dir} --dataset_name ${CASE_NAME} --train_feature_func default
cd ..
# Training
#-m LangSplat Output folder, three models will be generated (e.g. [wegs][vanillaCLIPExt][vanillaRecon]_1;[wegs][vanillaCLIPExt][vanillaRecon]_2;[wegs][vanillaCLIPExt][vanillaRecon]_3)
#-start_checkpoint: the 3DGS Reconstruction ckpt( 3DGS or WEGS)
python train.py -s ${scene_dir} -m ./output/${exp_name}/${CASE_NAME} --start_checkpoint ${scene_dir}/${reconstruction_case_name}/chkpnt${ckpt_iter}.pth --feature_level 1 --include_feature --resolution 2 --which_feature_fusion_func ${which_feature_fusion_func} --language_features_name language_features_dim3_${CASE_NAME} --iterations 60_000 
python train.py -s ${scene_dir} -m ./output/${exp_name}/${CASE_NAME} --start_checkpoint ${scene_dir}/${reconstruction_case_name}/chkpnt${ckpt_iter}.pth --feature_level 2 --include_feature --resolution 2 --which_feature_fusion_func ${which_feature_fusion_func} --language_features_name language_features_dim3_${CASE_NAME} --iterations 60_000
python train.py -s ${scene_dir} -m ./output/${exp_name}/${CASE_NAME} --start_checkpoint ${scene_dir}/${reconstruction_case_name}/chkpnt${ckpt_iter}.pth --feature_level 3 --include_feature --resolution 2 --which_feature_fusion_func ${which_feature_fusion_func} --language_features_name language_features_dim3_${CASE_NAME} --iterations 60_000 

# Render 
python render.py -s ${scene_dir} -m ./output/${exp_name}/${CASE_NAME}_1 --feature_level 1 --include_feature --resolution 2   --language_features_name language_features_dim3_${CASE_NAME} --which_feature_fusion_func ${which_feature_fusion_func} --skip_train --skip_test --render_small_batch
python render.py -s ${scene_dir} -m ./output/${exp_name}/${CASE_NAME}_2 --feature_level 2 --include_feature --resolution 2   --language_features_name language_features_dim3_${CASE_NAME} --which_feature_fusion_func ${which_feature_fusion_func} --skip_train --skip_test --render_small_batch
python render.py -s ${scene_dir} -m ./output/${exp_name}/${CASE_NAME}_3 --feature_level 3 --include_feature --resolution 2  --language_features_name language_features_dim3_${CASE_NAME} --which_feature_fusion_func ${which_feature_fusion_func} --skip_train --skip_test --render_small_batch

cd eval
which_post_feature_fusion_func='post_validMapLevel_avgImageWiseMaxValue|LocMax' # only for eval


gt_folder="/media/wangyz/DATA/UBUNTU_data/dataset/PT/label/brandenburg_gate"
root_path=".."

python evaluate_iou_loc_pt.py \
        --dataset_name ${CASE_NAME} \
        --feat_dir ${root_path}/output/${exp_name} \
        --ae_ckpt_dir ${root_path}/autoencoder/ckpt \
        --output_dir ${root_path}/eval_result \
        --mask_thresh 0.4 \
        --encoder_dims 256 128 64 32 3 \
        --decoder_dims 16 32 64 128 256 256 512 \
        --json_folder ${gt_folder} \
        --which_feature_fusion_func ${which_post_feature_fusion_func} \
        --sky_filter
cd ..
