#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import os
import pickle
import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)
            
        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
    def get_language_feature(self, language_feature_dir, feature_level,which_feature_fusion_func,num_aug_rendering,eval=False):
        if not eval:
            if which_feature_fusion_func == 'default':
                # only load default _s and _f
                language_feature_name = os.path.join(language_feature_dir, self.image_name)
                feature_path = language_feature_name + '_f.npy'
                segmentation_path = language_feature_name + '_s.npy'
                point_feature, mask = self.read_and_generate_feature_map_from_path_primitive(feature_path,
                                                                                             segmentation_path,
                                                                                             feature_level)
                t_mask = mask
                a_mask = mask
            elif which_feature_fusion_func == 'aug':

                point_feature_list = []
                mask_list = []
                language_feature_name = os.path.join(language_feature_dir, self.image_name)
                origin_feature_path = language_feature_name + '_f.npy'
                origin_segmentation_path = language_feature_name + '_s.npy'
                origin_point_feature, origin_mask = self.read_and_generate_feature_map_from_path_primitive(
                    origin_feature_path, origin_segmentation_path, feature_level)
                point_feature_list.append(origin_point_feature)
                mask_list.append(origin_mask)
                # load the num_aug_rendering
                # load _s;_s_ma_0,..._s_ma_3; load _f
                for i in range(num_aug_rendering):
                    aug_feature_path = os.path.join(language_feature_dir,
                                                    self.image_name + '_f_ma_' + str(i) + ".npy")
                    aug_segmentation_path = os.path.join(language_feature_dir,
                                                         self.image_name + '_s_ma_' + str(i) + ".npy")
                    aug_point_feature, aug_mask = self.read_and_generate_feature_map_from_path_primitive(aug_feature_path,
                                                                                                         aug_segmentation_path,
                                                                                                         feature_level)
                    point_feature_list.append(aug_point_feature)
                    mask_list.append(aug_mask)
                # fusion!
                point_feature = torch.stack(point_feature_list)
                point_feature = point_feature.view(-1, point_feature.shape[2], point_feature.shape[3])

                mask = torch.stack(mask_list).repeat(1,3,1,1).view(-1,mask_list[0].shape[1], mask_list[0].shape[2])
                a_mask, t_mask =  torch.zeros_like(mask), mask
            elif which_feature_fusion_func == "aug_wUncertainly_TM":
                point_feature_list = []
                mask_list = []
                language_feature_name = os.path.join(language_feature_dir, self.image_name)
                origin_feature_path = language_feature_name + '_f.npy'
                origin_segmentation_path = language_feature_name + '_s.npy'
                origin_point_feature, origin_mask = self.read_and_generate_feature_map_from_path_primitive(
                    origin_feature_path, origin_segmentation_path, feature_level)
                point_feature_list.append(origin_point_feature)
                mask_list.append(origin_mask)
                # load the num_aug_rendering
                # load _s;_s_ma_0,..._s_ma_3; load _f
                for i in range(num_aug_rendering):
                    aug_feature_path = os.path.join(language_feature_dir,
                                                    self.image_name + '_f_ma_' + str(i) + ".npy")
                    aug_segmentation_path = os.path.join(language_feature_dir,
                                                         self.image_name + '_s_ma_' + str(i) + ".npy")
                    aug_point_feature, aug_mask = self.read_and_generate_feature_map_from_path_primitive(aug_feature_path,
                                                                                                         aug_segmentation_path,
                                                                                                         feature_level)
                    point_feature_list.append(aug_point_feature)
                    mask_list.append(aug_mask)
                uncertainly_t_map_path = language_feature_name + '_uncertainly_map_T.npy'
                uncertainly_t_map = torch.from_numpy(np.load(uncertainly_t_map_path)) # [340,514] in R[0,1]
                y, x = torch.meshgrid(torch.arange(0, self.image_height), torch.arange(0, self.image_width))
                x = x.reshape(-1, 1)
                x = x.clip(0, uncertainly_t_map.shape[1] - 1)
                y = y.reshape(-1, 1)
                y = y.clip(0, uncertainly_t_map.shape[0] - 1)
                uncertainly_t_map = uncertainly_t_map[ y, x].squeeze(-1) #
                uncertainly_t_map = uncertainly_t_map.reshape(1, self.image_height, self.image_width) # origin uncertainly
                uncertainly_t_map = 1 - uncertainly_t_map
                # fusion!
                point_feature = torch.stack(point_feature_list)
                point_feature = point_feature.view(-1, point_feature.shape[2], point_feature.shape[3])
                mask = torch.stack(mask_list).repeat(1,3,1,1).view(-1,mask_list[0].shape[1], mask_list[0].shape[2]) #True or False-> become
                t_mask = mask * uncertainly_t_map
                a_mask, t_mask = torch.zeros_like(t_mask), t_mask
            # this fuc is used in the final exp
            elif which_feature_fusion_func == 'aug_wUncertainly_TMAM':
                point_feature_list = []
                mask_list = []
                language_feature_name = os.path.join(language_feature_dir, self.image_name)
                origin_feature_path = language_feature_name + '_f.npy'
                origin_segmentation_path = language_feature_name + '_s.npy'
                origin_point_feature, origin_mask = self.read_and_generate_feature_map_from_path_primitive(
                    origin_feature_path, origin_segmentation_path, feature_level)
                point_feature_list.append(origin_point_feature)
                mask_list.append(origin_mask)
                # load the num_aug_rendering
                # load _s;_s_ma_0,..._s_ma_3; load _f
                for i in range(num_aug_rendering):
                    aug_feature_path = os.path.join(language_feature_dir,
                                                    self.image_name + '_f_ma_' + str(i) + ".npy")
                    aug_segmentation_path = os.path.join(language_feature_dir,
                                                         self.image_name + '_s_ma_' + str(i) + ".npy")
                    aug_point_feature, aug_mask = self.read_and_generate_feature_map_from_path_primitive(aug_feature_path,
                                                                                                         aug_segmentation_path,
                                                                                                         feature_level)
                    point_feature_list.append(aug_point_feature)
                    mask_list.append(aug_mask)
                uncertainly_t_map_path = language_feature_name + '_uncertainly_map_T.npy'
                uncertainly_t_map = torch.from_numpy(np.load(uncertainly_t_map_path)) # [340,514] in R[0,1]
                y, x = torch.meshgrid(torch.arange(0, self.image_height), torch.arange(0, self.image_width))
                x = x.reshape(-1, 1)
                x = x.clip(0, uncertainly_t_map.shape[1] - 1)
                y = y.reshape(-1, 1)
                y = y.clip(0, uncertainly_t_map.shape[0] - 1)
                uncertainly_t_map = uncertainly_t_map[ y, x].squeeze(-1) #
                uncertainly_t_map = uncertainly_t_map.reshape(1, self.image_height, self.image_width) # origin uncertainly
                uncertainly_t_map = 1 - uncertainly_t_map
                # fusion!
                point_feature = torch.stack(point_feature_list)
                point_feature = point_feature.view(-1, point_feature.shape[2], point_feature.shape[3])
                mask = torch.stack(mask_list).repeat(1,3,1,1).view(-1,mask_list[0].shape[1], mask_list[0].shape[2]) #True or False-> become
                t_mask = mask * uncertainly_t_map

                uncertainly_a_map_path = language_feature_name + '_uncertainly_map_A.npy'
                uncertainly_a_map = torch.from_numpy(np.load(uncertainly_a_map_path)) # [340,514] in R[0,1]
                y, x = torch.meshgrid(torch.arange(0, self.image_height), torch.arange(0, self.image_width))
                x = x.reshape(-1, 1)
                x = x.clip(0, uncertainly_a_map.shape[1] - 1)
                y = y.reshape(-1, 1)
                y = y.clip(0, uncertainly_a_map.shape[0] - 1)
                uncertainly_a_map = uncertainly_a_map[ y, x].squeeze(-1) #
                uncertainly_a_map = uncertainly_a_map.reshape(1, self.image_height, self.image_width) # origin uncertainly
                uncertainly_a_map = 1 - uncertainly_a_map
                a_mask = mask * uncertainly_a_map * uncertainly_t_map
                # a_mask = mask * uncertainly_a_map
            else:
                raise NotImplementedError
            return point_feature.cuda(), t_mask.cuda(), a_mask.cuda()
        else:
            print('eval no gt ')
            return None, None, None


    def read_and_generate_feature_map_from_path_primitive(self,feature_path, segmentation_path,feature_level):
            # "xxx_f.npy";"xxx_s.npy"
            seg_map = torch.from_numpy(np.load(segmentation_path))  # [4,w,h]
            feature_map = torch.from_numpy(np.load(feature_path))  # [435,3]

            y, x = torch.meshgrid(torch.arange(0, self.image_height), torch.arange(0, self.image_width))
            x = x.reshape(-1, 1)
            x = x.clip(0, seg_map.shape[2] - 1)
            y = y.reshape(-1, 1)
            y = y.clip(0, seg_map.shape[1] - 1)
            seg = seg_map[:, y, x].squeeze(-1).long()  # [4,200670]
            mask = seg != -1
            if feature_level == 0:  # default
                point_feature1 = feature_map[seg[0:1]].squeeze(0)
                mask = mask[0:1].reshape(1, self.image_height, self.image_width)
            elif feature_level == 1:  # s
                point_feature1 = feature_map[seg[1:2]].squeeze(0)  # [200670,3]
                mask = mask[1:2].reshape(1, self.image_height, self.image_width)  # [1,390,530]
            elif feature_level == 2:  # m
                point_feature1 = feature_map[seg[2:3]].squeeze(0)
                mask = mask[2:3].reshape(1, self.image_height, self.image_width)
            elif feature_level == 3:  # l
                point_feature1 = feature_map[seg[3:4]].squeeze(0)
                mask = mask[3:4].reshape(1, self.image_height, self.image_width)
            else:
                raise ValueError("feature_level=", feature_level)
            # point_feature = torch.cat((point_feature2, point_feature3, point_feature4), dim=-1).to('cuda')
            point_feature = point_feature1.reshape(self.image_height, self.image_width, -1).permute(2, 0,
                                                                                                    1)  # [3,390,530]
            return point_feature, mask

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

