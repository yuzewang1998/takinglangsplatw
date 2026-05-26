# [TVCG 2026] Taking Language Embedded 3D Gaussian Splatting into the wild 
[Yuze Wang](https://yuzewang1998.github.io/), [Junyi Wang](https://junyiwang.github.io/),[Yue Qi](https://scse.buaa.edu.cn/info/1078/2661.htm)<br>| [Webpage](https://yuzewang1998.github.io/takinglangsplatw/) | <br>
| [Pre-trained Models](https://drive.google.com/drive/folders/1Ok64q8RyuqiBX62fLh2xVbOeyNg3IgQz) |<br>
| [Benchmark](https://drive.google.com/drive/folders/1Ok64q8RyuqiBX62fLh2xVbOeyNg3IgQz) |<br>

![Teaser image](assets/teaser.png)

This repository contains the official implementation associated with the paper **"Taking Language Embedded 3D Gaussian Splatting into the Wild"**. We also provide the **PT-OVS benchmark** and **pretrained models** for each scene.

## 🚀 Overview

The codebase consists of three main components:
- **Optimizer:** A PyTorch-based trainer that produces a MALE-GS model from SfM datasets with language feature inputs.
- **Scene-wise Autoencoder:** A module designed to alleviate the substantial memory demands of explicit high-dimensional modeling by compressing features.
- **PT-OVS Benchmark:** A specialized dataset for evaluating Open-Vocabulary Segmentation (OVS) in unconstrained "in-the-wild" environments.


The components have been tested on Ubuntu Linux 22.04. Instructions for setting up and running each of them are found in the sections below.

## 📊 Datasets

In the experiments section of our paper, we primarily utilized the proposed PT-OVS dataset.

The PT-OVS dataset is accessible for download via the following link: 

1. [Download Original PhotoTourism Dataset which contains RGB images, corresponding point cloud and camera poses](https://www.cs.ubc.ca/~kmyi/imw2020/data.html): 7 scenes in total (brandenburg_gate, buckingham_palace, notre_dame_front_facade, pantheon_exterior, taj_mahal, temple_nara_japan, trevi_fountain)

2. [Download our proposed PT-OVS Benchmark label](https://drive.google.com/drive/folders/1Ok64q8RyuqiBX62fLh2xVbOeyNg3IgQz), put it at the same level as the other scenes 

   

## 🔧 Installation

1. Cloning the Repository

Since the repository includes submodules, please clone it recursively:

```
# HTTPS
git clone --recursive https://github.com/yuzewang1998/takinglangsplatw.git
cd takinglangsplatw
```

2. Environment Setup

Our installation is based on Conda. We mainly follow the LangSplat environment setup. Before creating the environment, clone the language-feature preprocessing dependency expected by `environment.yml`:

```shell
git clone https://github.com/minghanqin/segment-anything-langsplat.git submodules/segment-anything-langsplat
conda env create --file environment.yml
conda activate malegs
```

**Note:** `environment.yml` installs the CUDA rasterization, KNN, and language-feature preprocessing modules used by this repository. Please also download the SAM checkpoints to `ckpts/` from the [official Segment Anything repository](https://github.com/facebookresearch/segment-anything).


 3. Hardware Requirements

- CUDA-ready GPU with Compute Capability 7.0+
- 24 GB VRAM (to train to paper evaluation quality)

## QuickStart

Download the pretrained model, containing constructed WE-GS models, trained autoencoder ckpt, and trained MALE-GS ckpts for a specific scene, and you can evaluate the method.

```shell
PYTHONPATH=. python eval/evaluate_iou_loc_pt.py \
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
```
## Pipeline
- **Step 1: Train the radiance field.**

  You can use an arbitrary 3DGS-based radiance-field reconstruction method. We have tested vanilla 3DGS, GS-W, and WE-GS; more advanced in-the-wild reconstruction methods can lead to more accurate 3D OVS results. This repository does not include the WE-GS training code, so first prepare a reconstructed radiance field externally (for example with WE-GS) or use the reconstruction checkpoints provided with our pretrained models.

  The reconstructed model should then be placed under the corresponding PT scene folder and referenced by `--itw_model_path` / `--start_checkpoint` in the following steps.

- **Step 2: Generate language features and uncertainty maps for the scenes.**

  Modify the paths for `--dataset_path`, `--iteration`, `--itw_sh_degree`, `--itw_source_path`, and `--itw_model_path` in `bash_preprocess.sh` to match your reconstructed radiance field.

  ```shell
  bash bash_preprocess.sh
  ```

  Because unconstrained photo collections can contain many images, this step may take time. For a fast test, we recommend using our provided checkpoints.

- **Step 3: Train the uncertainty-aware Autoencoder and get the lower-dimensional features.**

   You can refer to train_brand.sh to input the arguments.

  ```shell
  # train the autoencoder
  cd autoencoder
  python train.py --dataset_path ${scene_dir} --dataset_name ${CASE_NAME} --train_feature_func default --num_epochs 100 --train_with_uncertainly_map --fusion_uncertainly_map_func direct_multiply

  # get the compressed language feature of the scene
  python test.py --dataset_path ${scene_dir} --dataset_name ${CASE_NAME} --train_feature_func default
  cd ..
  ```

  Our model expects the following dataset structure in the source path location, similar to MALE-GS:
  ```
  <dataset_name>
  |---images
  |   |---<image 0>
  |   |---<image 1>
  |   |---...
  |---language_features
  |   |---00_f.npy
  |   |---00_s.npy
  |   |---...
  |---language_features_dim3_<dataset_name>
  |   |---00_f.npy
  |   |---00_s.npy
  |   |---...
  |---output
  |   |---<dataset_name>
  |   |   |---point_cloud/iteration_30000/point_cloud.ply
  |   |   |---cameras.json
  |   |   |---cfg_args
  |   |   |---chkpnt30000.pth
  |   |   |---input.ply
  |---sparse
      |---0
          |---cameras.bin
          |---images.bin
          |---points3D.bin
  ```

- **Step 4: Train the MALE-GS.**

  ​	You can refer to train_brand.sh to input the arguments.

  ```shell
  python train.py -s ${scene_dir} -m ./output/${exp_name}/${CASE_NAME} --start_checkpoint ${scene_dir}/${reconstruction_case_name}/chkpnt${ckpt_iter}.pth --feature_level 1 --include_feature --resolution 2 --which_feature_fusion_func ${which_feature_fusion_func} --language_features_name language_features_dim3_${CASE_NAME} --iterations 30000
  ```

- **Step 5: Render the MALE-GS.**

  ```shell
  python render.py -s ${scene_dir} -m ./output/${exp_name}/${CASE_NAME}_1 --feature_level 1 --include_feature --resolution 2 --language_features_name language_features_dim3_${CASE_NAME} --which_feature_fusion_func ${which_feature_fusion_func} --skip_train --skip_test --render_small_batch
  ```

- **Step 6: Eval.**
  Evaluate the performance on the PT-OVS benchmark. You can refer to train_brand.sh to input the arguments.

  ```shell
  PYTHONPATH=. python eval/evaluate_iou_loc_pt.py \
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
  ```

