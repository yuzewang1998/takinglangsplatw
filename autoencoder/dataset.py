import os
import glob
import numpy as np
import torch
from huggingface_hub import dataset_info
from pygments.lexer import default
from torch.utils.data import Dataset
def cauc_mean_uncertainly_map(s,f,uncertainly_map):
    N = f.shape[0]

    mean_uncertain_probs = np.zeros((N,1))
    for i in range(N):
        mask = (s == i).any(axis=0)
        if mask.sum()> 0 :
                mean_uncertain_probs[i] = uncertainly_map[mask].mean()
        else:
            mean_uncertain_probs[i] = 0.0
    torch.from_numpy(mean_uncertain_probs)
    return mean_uncertain_probs
class Autoencoder_dataset(Dataset):
    def __init__(self, data_dir,train_with_uncertainly_map,train_feature_func,is_training = True):
        self.train_with_uncertainly_map = train_with_uncertainly_map
        self.train_feature_func = train_feature_func
        self.data_dic = {}
        if self.train_feature_func == 'default':
            if is_training:
                data_names = glob.glob(os.path.join(data_dir, '*f.npy'))
            else:
                data_names = glob.glob(os.path.join(data_dir, '*f*.npy'))
            for i in range(len(data_names)):
                features = np.load(data_names[i])
                name = data_names[i].split('/')[-1].split('.')[0]# xxxx_f
                self.data_dic[name] = features.shape[0]
                if i == 0:
                    data = features
                else:
                    data = np.concatenate([data, features], axis=0) #[N,512]
        elif self.train_feature_func == 'simple_aug':
            data_names = glob.glob(os.path.join(data_dir, '*f*.npy'))
            for i in range(len(data_names)):
                features = np.load(data_names[i])
                name = data_names[i].split('/')[-1].split('.')[0]
                self.data_dic[name] = features.shape[0]
                if i == 0:
                    data = features
                else:
                    data = np.concatenate([data, features], axis=0)
        elif self.train_feature_func == 'train_fused_avg':
            # 这个可以被废除，因为这样太占用内存了。读那么dense的图片（W，H，512），区区64G内存没法接受；并且这个预处理太慢了。in-the-wild很难接受
            data_names = glob.glob(os.path.join(data_dir, '*f.npy'))

            len_num_aug = len(glob.glob(data_names[0].split('.')[0][:-2]+"_f*.npy"))-2 # -2 : origin_image;origin_image_appearance_rendering
            for idx in range(len(data_names)):
                dense_feat_map_list = []
                language_feature_name = data_names[idx].split('.')[0][:-2]
                origin_feature_path = data_names[idx] # xxx_f.npy
                origin_segmentation_path = language_feature_name + '_s.npy'
                feature = np.load(origin_feature_path)
                segment = np.load(origin_segmentation_path)
                dense_feat_map = feature[segment]
                dense_feat_map_list.append(dense_feat_map)
                #Aug feature
                for aug_idx in range(len_num_aug):
                    aug_language_feature_path = language_feature_name + "_f_ma_" + str(aug_idx) + ".npy"
                    aug_segmentation_path = language_feature_name + "_s_ma_"+ str(aug_idx) + ".npy"
                    feature = np.load(aug_language_feature_path)
                    segment = np.load(aug_segmentation_path)
                    dense_feat_map = feature[segment]
                    dense_feat_map_list.append(dense_feat_map)
                # origin_segmentation_path = language_feature_name + '_s.npy'
                avg_feature = np.stack(dense_feat_map_list).mean(axis=0).reshape(-1,512)
                if idx == 0:
                    data = avg_feature
                else:
                    data = np.concatenate([data, avg_feature], axis=0)
                print('processing:',idx)
        else:
            raise NotImplementedError


        self.data = data

        if train_with_uncertainly_map:
            self.uncertain_probs = []
            uncertainly_map_names = [os.path.join(data_dir,data_name.split('/')[-1].split('.')[0].split('f')[0]+"uncertainly_map_T.npy") for data_name in data_names]
            seg_map_names = [os.path.join(data_dir,data_name.split('/')[-1].split('.')[0].split('f')[0]+"s.npy") for data_name in data_names]
            for i in range(len(uncertainly_map_names)):
                print(uncertainly_map_names[i])
                uncertainly_map = np.load(uncertainly_map_names[i])
                features = np.load(data_names[i])
                seg_map = np.load(seg_map_names[i])
                uncertain_prob = cauc_mean_uncertainly_map(seg_map,features,uncertainly_map)
                if i ==0:
                    uncertain_probs = uncertain_prob
                else:
                    uncertain_probs = np.concatenate([uncertain_probs, uncertain_prob], axis=0)
            self.uncertain_probs = uncertain_probs
        print('done for prepare data.')
    def __getitem__(self, index):
        data = torch.tensor(self.data[index])
        if self.train_with_uncertainly_map:
            uncertain_probs = torch.tensor(self.uncertain_probs[index])
        else:
            # return None
            uncertain_probs = torch.zeros_like(data)
        return data,uncertain_probs

    def __len__(self):
        return self.data.shape[0] 