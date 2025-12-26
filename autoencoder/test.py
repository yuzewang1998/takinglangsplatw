import os
import numpy as np
import torch
import argparse
import shutil
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import Autoencoder_dataset
from model import Autoencoder
import fnmatch
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--encoder_dims',
                    nargs = '+',
                    type=int,
                    default=[256, 128, 64, 32, 3],
                    )
    parser.add_argument('--decoder_dims',
                    nargs = '+',
                    type=int,
                    default=[16, 32, 64, 128, 256, 256, 512],
                    )
    parser.add_argument('--train_with_uncertainly_map', action='store_true')
    parser.add_argument('--train_feature_func',type=str,choices=['default','simple_aug','train_fused_avg'], default = 'default')
    args = parser.parse_args()
    
    dataset_name = args.dataset_name
    encoder_hidden_dims = args.encoder_dims
    decoder_hidden_dims = args.decoder_dims
    dataset_path = args.dataset_path
    ckpt_path = f"ckpt/{dataset_name}/best_ckpt.pth"

    data_dir = f"{dataset_path}/language_features"
    output_dir = f"{dataset_path}/language_features_dim3_{dataset_name}"
    os.makedirs(output_dir, exist_ok=True)

    print('[START] Testing the Auto Encoder')
    # copy the segmentation map
    for filename in os.listdir(data_dir):
        if fnmatch.fnmatch(filename, "*_s*.npy"):
            source_path = os.path.join(data_dir, filename)
            target_path = os.path.join(output_dir, filename)
            shutil.copy(source_path, target_path)
    # copy the uncertainly map
    for filename in os.listdir(data_dir):
        if fnmatch.fnmatch(filename, "*_uncertainly_map*.npy"):
            source_path = os.path.join(data_dir, filename)
            target_path = os.path.join(output_dir, filename)
            shutil.copy(source_path, target_path)

    checkpoint = torch.load(ckpt_path)
    train_dataset = Autoencoder_dataset(data_dir,args.train_with_uncertainly_map,args.train_feature_func,is_training=False)

    test_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=256,
        shuffle=False, 
        num_workers=16, 
        drop_last=False   
    )


    model = Autoencoder(encoder_hidden_dims, decoder_hidden_dims).to("cuda:0")

    model.load_state_dict(checkpoint)
    model.eval()

    for idx, feature in tqdm(enumerate(test_loader)):
        feature, _ = feature
        data = feature.to("cuda:0")
        with torch.no_grad():
            outputs = model.encode(data).to("cpu").numpy()  
        if idx == 0:
            features = outputs
        else:
            features = np.concatenate([features, outputs], axis=0)

    os.makedirs(output_dir, exist_ok=True)
    start = 0
    
    for k,v in train_dataset.data_dic.items():
        path = os.path.join(output_dir, k)
        np.save(path, features[start:start+v])
        start += v
    print('[END] Testing the Auto Encoder')