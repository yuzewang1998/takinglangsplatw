#!/usr/bin/env python
from __future__ import annotations

import json
import os
import glob
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Union
from argparse import ArgumentParser
import logging
import cv2
import numpy as np
import torch
import time

from sympy.codegen import Print
from sympy.logic.inference import valid
from tqdm import tqdm

import sys

sys.path.append("..")
import colormaps
from autoencoder.model import Autoencoder
from openclip_encoder import OpenCLIPNetwork
from utils import smooth, colormap_saving, vis_mask_save, polygon_to_mask, stack_mask, show_result


# user

def get_logger(name, log_file=None, log_level=logging.INFO, file_mode='w'):
    logger = logging.getLogger(name)
    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]

    if log_file is not None:
        file_handler = logging.FileHandler(log_file, file_mode)
        handlers.append(file_handler)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)
    logger.setLevel(log_level)
    return logger


def eval_gt_lerfdata(json_folder: Union[str, Path] = None, ouput_path: Path = None, resolution:int = 1) -> Dict:
    """
    organise lerf's gt annotations
    gt format:
        file name: frame_xxxxx.json
        file content: labelme format
    return:
        gt_ann: dict()
            keys: str(int(idx))
            values: dict()
                keys: str(label)
                values: dict() which contain 'bboxes' and 'mask' bbox ndarry (4; mask ndarry (730,988)
    """
    gt_json_paths = sorted(glob.glob(os.path.join(str(json_folder), '*.json')))
    img_paths = sorted(glob.glob(os.path.join(str(json_folder), '*.jpg'))) + sorted(glob.glob(os.path.join(str(json_folder), '*.JPG')))
    gt_ann = {}
    for js_path in gt_json_paths:
        img_ann = defaultdict(dict)
        with open(js_path, 'r') as f:
            gt_data = json.load(f)

        h, w = gt_data['info']['height'], gt_data['info']['width']
        # idx = int(gt_data['info']['name'].split('_')[-1].split('.jpg')[0]) - 1
        idx = gt_data['info']['name'][:-4]
        for prompt_data in gt_data["objects"]:
            label = prompt_data['category']  # str
            box = np.asarray(prompt_data['bbox']).reshape(-1)  # ndarry(4,)         # x1y1x2y2
            box = box // resolution
            mask = polygon_to_mask((h, w), prompt_data['segmentation'])  # ndarry(730,988)
            if img_ann[label].get('mask', None) is not None:
                mask = stack_mask(img_ann[label]['mask'], mask)
                img_ann[label]['bboxes'] = np.concatenate(
                    [img_ann[label]['bboxes'].reshape(-1, 4), box.reshape(-1, 4)], axis=0)
            else:
                img_ann[label]['bboxes'] = box
            img_ann[label]['mask'] = mask
            img_ann['h'] = h
            img_ann['w'] = w
            # # save for visulsization
            save_path = ouput_path / 'gt' / gt_data['info']['name'].split('.jpg')[0] / f'{label}.jpg'
            save_path.parent.mkdir(exist_ok=True, parents=True)
            vis_mask_save(mask, save_path)
        gt_ann[f'{idx}'] = img_ann

    return gt_ann, img_paths  # img_paths:list:6


def activate_stream(sem_map,
                    image,
                    clip_model,
                    image_name: Path = None,
                    img_ann: Dict = None,
                    thresh: float = 0.5,
                    colormap_options=None,resolution=1):
    valid_map = clip_model.get_max_across(sem_map)  # 3xkx832x1264
    n_head, n_prompt, h, w = valid_map.shape

    # positive prompts
    chosen_iou_list, chosen_lvl_list = [], []
    for k in range(n_prompt):
        iou_lvl = np.zeros(n_head)
        mask_lvl = np.zeros((n_head, h, w))
        for i in range(n_head):
            # NOTE 加滤波结果后的激活值图中找最大值点
            scale = 30
            kernel = np.ones((scale, scale)) / (scale ** 2)
            np_relev = valid_map[i][k].cpu().numpy()
            avg_filtered = cv2.filter2D(np_relev, -1, kernel)
            avg_filtered = torch.from_numpy(avg_filtered).to(valid_map.device)
            valid_map[i][k] = 0.5 * (avg_filtered + valid_map[i][k])

            output_path_relev = image_name / 'heatmap' / f'{clip_model.positives[k]}_{i}'
            output_path_relev.parent.mkdir(exist_ok=True, parents=True)
            colormap_saving(valid_map[i][k].unsqueeze(-1), colormap_options,
                            output_path_relev)

            # NOTE 与lerf一致，激活值低于0.5的认为是背景
            p_i = torch.clip(valid_map[i][k] - 0.5, 0, 1).unsqueeze(-1)
            valid_composited = colormaps.apply_colormap(p_i / (p_i.max() + 1e-6), colormaps.ColormapOptions("turbo"))
            mask = (valid_map[i][k] < 0.5).squeeze()
            image = torch.Tensor(cv2.resize(image.cpu().numpy(), (mask.shape[1], mask.shape[0]))).cuda()
            valid_composited[mask, :] = image[mask, :] * 0.3
            output_path_compo = image_name / 'composited' / f'{clip_model.positives[k]}_{i}'
            output_path_compo.parent.mkdir(exist_ok=True, parents=True)
            colormap_saving(valid_composited, colormap_options, output_path_compo)

            # truncate the heatmap into mask/home/wangyz/Documents/Nut/我的坚果云/0近期资料/1_LangSplat-W/manuscript_LangSplat-W-ISMAR-2025-/figures/home/wangyz/Documents/Nut/我的坚果云/0近期资料/1_LangSplat-W/manuscript_LangSplat-W-ISMAR-2025-/figures
            output = valid_map[i][k]
            output = output - torch.min(output)
            output = output / (torch.max(output) + 1e-9)
            output = output * (1.0 - (-1.0)) + (-1.0)
            output = torch.clip(output, 0, 1)

            mask_pred = (output.cpu().numpy() > thresh).astype(np.uint8)
            mask_pred = smooth(mask_pred)
            mask_lvl[i] = mask_pred
            mask_gt = img_ann[clip_model.positives[k]]['mask'].astype(np.uint8)

            mask_gt = cv2.resize(mask_gt, ( mask_pred.shape[1], mask_pred.shape[0]))
            # calculate iou
            intersection = np.sum(np.logical_and(mask_gt, mask_pred))
            union = np.sum(np.logical_or(mask_gt, mask_pred))
            iou = np.sum(intersection) / np.sum(union)
            iou_lvl[i] = iou

        score_lvl = torch.zeros((n_head,), device=valid_map.device)
        for i in range(n_head):
            score = valid_map[i, k].max()
            score_lvl[i] = score
        chosen_lvl = torch.argmax(score_lvl)

        chosen_iou_list.append(iou_lvl[chosen_lvl])
        chosen_lvl_list.append(chosen_lvl.cpu().numpy())

        # save for visulsization
        save_path = image_name / f'chosen_{clip_model.positives[k]}.png'
        vis_mask_save(mask_lvl[chosen_lvl], save_path)

    return chosen_iou_list, chosen_lvl_list

def activate_stream_post(sem_map,
                    image,
                    clip_model,
                    image_name: Path = None,
                    img_ann: Dict = None,
                    thresh: float = 0.5,
                    colormap_options=None,
                    resolution=1,
                    which_feature_fusion_func = 'post_validMapLevel_avg',is_sky_filter = False):
    '''

    Args: which_feature_fusion_func :
        First item, it must begin with post_, or Bug
        Second item, decide which step to fusion the feature , choices =  ['validMap_level','result_level(vote)',...]
        Third item, deside which fusion fuction , choices = ['avg','max','min', ...]

      '''
    # The difference with activate_stream() is the input sem_map is a list, such as [N,3]；
    # sem_map:[4,3,528,390,512]
    #image_[528,390,3]
    assert which_feature_fusion_func.startswith('post_'), 'Error, It must a bug, no codition without start with post_ can be here'
    _, args_fusion_level, args_fusion_opt = which_feature_fusion_func.split('_')
    valid_map = clip_model.get_max_across_post(sem_map,is_sky_filter)  # [3(lvl),k_text_prompt,W,H]->[4,3,k_text_prompt_W,H]
    # Attention: if is_sky_filter TRUE, the size of valid_map will be [3,k_text_prompt + 1, W, H], the last valid map will be the sky valid map
    if  is_sky_filter:
        sky_valid_map = valid_map[:,:,-1,...] #[4,3,528,390]
        valid_map = valid_map[:,:,:-1,...] #[4,3,2,528,390]
    n_var, n_head, n_prompt, h, w = valid_map.shape # valid_map [4,3,2,528,390]

    # head: lvl
    # positive prompts
    chosen_iou_list, chosen_lvl_list = [], []
    if args_fusion_level == 'validMapLevel':
        # Fusion the valid_map from [4(n_var), 3(lvl), n_text_prompt, W, H] to [3(lvl), n_text_prompt, W, H] first.
        if args_fusion_opt == 'avg': # ablation 9
            valid_map = torch.mean(valid_map,dim=0) # [4(n_var), 3(lvl), n_text_prompt, W, H] -> [3(lvl), n_text_prompt, W, H]
        elif args_fusion_opt == 'max':#ablation 8
            valid_map = torch.max(valid_map, dim=0)[0] # [4(n_var), 3(lvl), n_text_prompt, W, H]-> [3(lvl), n_text_prompt, W, H]
        elif args_fusion_opt == 'avgPixelwiseNormalDistribution': #ablation 10
            # Generate a probablity map for each n_var: weight:[4,3]
            mean_valid_map = valid_map.mean(dim=0)[None,...].repeat(valid_map.shape[0],1,1,1,1)
            weight = 1 / (torch.square(mean_valid_map - valid_map)+ 1e-6)
            #weight = torch.clamp(weight, min=0, max=10) # ...good this make everything 0.25. it be average
            sum_weight = torch.sum(weight,dim=0,keepdim=True)
            normalized_weight = weight/ sum_weight
            valid_map = (normalized_weight * valid_map).sum(dim=0)
        elif args_fusion_opt == 'avgImageWiseNormalDistribution':
            mean_valid_map = valid_map.mean(dim=-1).mean(dim=-1) #[4,3,2]
            mean_mean_valid_map = mean_valid_map.mean(dim=0)[None,...]
            weight = 1 / (torch.square(mean_mean_valid_map - mean_valid_map)+ 1e-6) #[4,3,2]
            #weight = torch.clamp(weight, min=0, max=10) # ...good
            sum_weight = torch.sum(weight,dim=0,keepdim=True)
            normalized_weight = weight/ sum_weight#[4,3,2]
            normalized_weight = normalized_weight[...,None,None]
            valid_map = (normalized_weight * valid_map).sum(dim=0)
        elif args_fusion_opt == 'avgImageWiseMaxValue':
            # fusion according to the max value of the pixel
            weight = valid_map.max(dim=-1)[0].max(dim=-1)[0]
            sum_weight = torch.sum(weight,dim=0,keepdim=True)
            normalized_weight = weight/ sum_weight#[4,3,2]
            normalized_weight = normalized_weight[...,None,None]
            valid_map = (normalized_weight * valid_map).sum(dim=0)
        elif args_fusion_opt == 'avgHybridImageWiseMaxValuePixelWiseMaxValue':
            image_wise_weight = valid_map.max(dim=-1)[0].max(dim=-1)[0][...,None,None] + 1e-6 # [4,3,2,1,1]
            mean_valid_map = valid_map.mean(dim=0)[None, ...].repeat(valid_map.shape[0], 1, 1, 1, 1)
            pixel_wise_weight = 1 / (torch.square(mean_valid_map - valid_map) + 1e-6) # [4,3,2,w,h]
            weight = image_wise_weight * pixel_wise_weight
            sum_weight = torch.sum(weight,dim=0,keepdim=True)
            normalized_weight = weight/ sum_weight
            valid_map = (normalized_weight * valid_map).sum(dim=0)
        else:
            raise  NotImplementedError
        if is_sky_filter:
            # sky_valid_map #[4,3,528,390]
            # valid_map #[3,2,w,h]
            # i suppose nobody want to segment the sky.... if seg the sky... just use the last channel as the valid map
            # First: fuse the sky. Every view i think will seg the same sky
            sky_valid_map = sky_valid_map.view(-1,sky_valid_map.shape[-2],sky_valid_map.shape[-1])
            # sky_valid_map = sky_valid_map[0]
            # sky_valid_map = torch.mean(sky_valid_map,dim=0) # [528,390]
            sky_valid_map = torch.max(sky_valid_map,dim=0)[0]
            # scale = 30
            # kernel = np.ones((scale, scale)) / (scale ** 2)
            # np_relev = sky_valid_map.cpu().numpy()
            # avg_filtered = cv2.filter2D(np_relev, -1, kernel)
            # avg_filtered = torch.from_numpy(avg_filtered).to(sky_valid_map.device)
            # sky_valid_map = 0.5 * (avg_filtered + sky_valid_map)
            # output_sky = sky_valid_map
            # output_sky = output_sky - torch.min(output_sky)
            # output_sky = output_sky / (torch.max(output_sky) + 1e-9)
            # output_sky = output_sky * (1.0 - (-1.0)) + (-1.0)
            # output_sky = torch.clip(output_sky, 0, 1)
            # not_sky_thresh = 0.8
            # mask_not_sky_pred = (output_sky.cpu().numpy() <not_sky_thresh).astype(np.uint8)
            # import torchvision
            # save_vis_mask = torch.from_numpy(mask_not_sky_pred)[None,...].repeat(3,1,1).to(torch.float32)
            # torchvision.utils.save_image(save_vis_mask,'/home/wangyz/Downloads/{}.jpg'.format(time.time()))
            # sky_valid_map = torch.max(sky_valid_map,dim=0)[0]
        for k in range(n_prompt):
            iou_lvl = np.zeros(n_head)
            mask_lvl = np.zeros((n_head, h, w))
            for i in range(n_head):
                # NOTE 加滤波结果后的激活值图中找最大值点
                scale = 30
                kernel = np.ones((scale, scale)) / (scale ** 2)
                if is_sky_filter:
                    select_output = (1 - sky_valid_map) * valid_map[i][k]
                    np_relev = select_output.cpu().numpy()
                else:
                    np_relev = valid_map[i][k].cpu().numpy()
                avg_filtered = cv2.filter2D(np_relev, -1, kernel)
                avg_filtered = torch.from_numpy(avg_filtered).to(valid_map.device)
                valid_map[i][k] = 0.5 * (avg_filtered + valid_map[i][k])

                output_path_relev = image_name / 'heatmap' / f'{clip_model.positives[k]}_{i}'
                output_path_relev.parent.mkdir(exist_ok=True, parents=True)
                colormap_saving(valid_map[i][k].unsqueeze(-1), colormap_options,
                                output_path_relev)

                # NOTE 与lerf一致，激活值低于0.5的认为是背景
                # p_i = valid_map[i][k]
                # p_i = p_i - torch.min(p_i)
                # p_i = p_i / (torch.max(p_i) + 1e-9)
                # p_i = p_i * (1.0 - (-1.0)) + (-1.0)
                # p_i = torch.clip(p_i, 0, 1).unsqueeze(-1)
                p_i = torch.clip(valid_map[i][k] - 0.5, 0, 1).unsqueeze(-1)
                p_i = p_i / (p_i.max() + 1e-6)
                valid_composited = colormaps.apply_colormap(p_i,
                                                            colormaps.ColormapOptions("turbo"))
                mask = (valid_map[i][k] < 0.5).squeeze() #[W,h]
                image = torch.Tensor(cv2.resize(image.cpu().numpy(), (mask.shape[1], mask.shape[0]))).cuda()
                valid_composited[mask, :] = image[mask, :] * 0.3
                output_path_compo = image_name / 'composited' / f'{clip_model.positives[k]}_{i}'
                output_path_compo.parent.mkdir(exist_ok=True, parents=True)
                colormap_saving(valid_composited, colormap_options, output_path_compo)

                # truncate the heatmap into mask
                output = valid_map[i][k]
                output = output - torch.min(output)
                output = output / (torch.max(output) + 1e-9)
                output = output * (1.0 - (-1.0)) + (-1.0)
                output = torch.clip(output, 0, 1)

                mask_pred = (output.cpu().numpy() > thresh).astype(np.uint8)
                # if is_sky_filter:
                #     mask_pred = np.logical_and(mask_pred, mask_not_sky_pred)
                mask_pred = smooth(mask_pred)
                mask_lvl[i] = mask_pred
                mask_gt = img_ann[clip_model.positives[k]]['mask'].astype(np.uint8)

                mask_gt = cv2.resize(mask_gt, (mask_pred.shape[1], mask_pred.shape[0]))
                # calculate iou
                intersection = np.sum(np.logical_and(mask_gt, mask_pred))
                union = np.sum(np.logical_or(mask_gt, mask_pred))
                iou = np.sum(intersection) / np.sum(union)
                iou_lvl[i] = iou
            # 在这里按照n_head来评价score_lvl；并且按照最大iou来选择

            score_lvl = torch.zeros((n_head,), device=valid_map.device)
            for i in range(n_head):
                score = valid_map[i, k].max()
                score_lvl[i] = score
            chosen_lvl = torch.argmax(score_lvl)

            chosen_iou_list.append(iou_lvl[chosen_lvl])
            chosen_lvl_list.append(chosen_lvl.cpu().numpy())

            # save for visulsization
            save_path = image_name / f'chosen_{clip_model.positives[k]}.png'
            vis_mask_save(mask_lvl[chosen_lvl], save_path)
    elif args_fusion_level == 'resultLevel':
        # In this fusion level, (n_var x n_lvl) valid map will be cauculate, and fusion with the final outputs.
        # valid_map [4,3,2,528,390]->[12,2,528,390]
        n_head = n_var * n_head # n_head : 3 x 4
        valid_map = valid_map.view(n_head, n_prompt, h, w)
        for k in range(n_prompt):
            iou_lvl = np.zeros(n_head)
            mask_lvl = np.zeros((n_head, h, w))
            for i in range(n_head):
                # NOTE 加滤波结果后的激活值图中找最大值点
                scale = 30
                kernel = np.ones((scale, scale)) / (scale ** 2)
                np_relev = valid_map[i][k].cpu().numpy()
                avg_filtered = cv2.filter2D(np_relev, -1, kernel)
                avg_filtered = torch.from_numpy(avg_filtered).to(valid_map.device)
                valid_map[i][k] = 0.5 * (avg_filtered + valid_map[i][k])
                output_path_relev = image_name / 'heatmap' / f'{clip_model.positives[k]}_{i}'
                output_path_relev.parent.mkdir(exist_ok=True, parents=True)
                colormap_saving(valid_map[i][k].unsqueeze(-1), colormap_options,
                                output_path_relev)
                # NOTE 与lerf一致，激活值低于0.5的认为是背景
                p_i = torch.clip(valid_map[i][k] - 0.5, 0, 1).unsqueeze(-1)
                valid_composited = colormaps.apply_colormap(p_i / (p_i.max() + 1e-6),
                                                            colormaps.ColormapOptions("turbo"))
                mask = (valid_map[i][k] < 0.5).squeeze()
                image = torch.Tensor(cv2.resize(image.cpu().numpy(), (mask.shape[1], mask.shape[0]))).cuda()
                valid_composited[mask, :] = image[mask, :] * 0.3
                output_path_compo = image_name / 'composited' / f'{clip_model.positives[k]}_{i}'
                output_path_compo.parent.mkdir(exist_ok=True, parents=True)
                colormap_saving(valid_composited, colormap_options, output_path_compo)
                # truncate the heatmap into mask
                output = valid_map[i][k]
                output = output - torch.min(output)
                output = output / (torch.max(output) + 1e-9)
                output = output * (1.0 - (-1.0)) + (-1.0)
                output = torch.clip(output, 0, 1)

                mask_pred = (output.cpu().numpy() > thresh).astype(np.uint8)
                mask_pred = smooth(mask_pred)
                mask_lvl[i] = mask_pred
                mask_gt = img_ann[clip_model.positives[k]]['mask'].astype(np.uint8)

                mask_gt = cv2.resize(mask_gt, (mask_pred.shape[1], mask_pred.shape[0]))
                # calculate iou
                intersection = np.sum(np.logical_and(mask_gt, mask_pred))
                union = np.sum(np.logical_or(mask_gt, mask_pred))
                iou = np.sum(intersection) / np.sum(union)
                iou_lvl[i] = iou
            # fusion the final result at here, actually, not a fusion, but more than a decision or chosen
            score_lvl = torch.zeros((n_head,), device=valid_map.device)
            if args_fusion_opt == 'globalMax':
                for i in range(n_head):
                    score = valid_map[i, k].max() # valid_map [12,2,528,390] i->which image; k->which text prompt; score:for a specific image i; text prompt k,cauculate the max pixel of the whole image
                    score_lvl[i] = score
                chosen_lvl = torch.argmax(score_lvl)
            else:
                raise NotImplementedError
            chosen_iou_list.append(iou_lvl[chosen_lvl])
            chosen_lvl_list.append(chosen_lvl.cpu().numpy())

            # save for visulsization
            save_path = image_name / f'chosen_{clip_model.positives[k]}.png'
            vis_mask_save(mask_lvl[chosen_lvl], save_path)
    else:
        raise NotImplementedError

    return chosen_iou_list, chosen_lvl_list


def lerf_localization(sem_map, image, clip_model, image_name, img_ann):
    output_path_loca = image_name / 'localization'
    output_path_loca.mkdir(exist_ok=True, parents=True)
    valid_map = clip_model.get_max_across(sem_map)  # 3xkx832x1264
    n_head, n_prompt, h, w = valid_map.shape
    # positive prompts
    acc_num = 0
    keys = [key for key in img_ann.keys() if key not in ['h', 'w']]
    positives = list(keys)
    for k in range(len(positives)):
        select_output = valid_map[:, k]

        # NOTE 平滑后的激活值图中找最大值点
        scale = 30
        kernel = np.ones((scale, scale)) / (scale ** 2)
        np_relev = select_output.cpu().numpy()
        avg_filtered = cv2.filter2D(np_relev.transpose(1, 2, 0), -1, kernel)

        score_lvl = np.zeros((n_head,))
        coord_lvl = []
        for i in range(n_head):
            score = avg_filtered[..., i].max()
            coord = np.nonzero(avg_filtered[..., i] == score)
            score_lvl[i] = score
            coord_lvl.append(np.asarray(coord).transpose(1, 0)[..., ::-1])

        selec_head = np.argmax(score_lvl)
        coord_final = coord_lvl[selec_head]

        for box in img_ann[positives[k]]['bboxes'].reshape(-1, 4):
            flag = 0
            x1, y1, x2, y2 = box
            x_min, x_max = min(x1, x2), max(x1, x2)
            y_min, y_max = min(y1, y2), max(y1, y2)
            for cord_list in coord_final:
                if (cord_list[0] >= x_min and cord_list[0] <= x_max and
                        cord_list[1] >= y_min and cord_list[1] <= y_max):
                    acc_num += 1
                    flag = 1
                    break
            if flag != 0:
                break

        # NOTE 将平均后的结果与原结果相加，抑制噪声并保持激活边界清晰
        avg_filtered = torch.from_numpy(avg_filtered[..., selec_head]).unsqueeze(-1).to(select_output.device)
        torch_relev = 0.5 * (avg_filtered + select_output[selec_head].unsqueeze(-1))
        p_i = torch.clip(torch_relev - 0.5, 0, 1)
        valid_composited = colormaps.apply_colormap(p_i / (p_i.max() + 1e-6), colormaps.ColormapOptions("turbo"))
        mask = (torch_relev < 0.5).squeeze()
        image = torch.Tensor(cv2.resize(image.cpu().numpy(), (mask.shape[1], mask.shape[0]))).cuda()
        valid_composited[mask, :] = image[mask, :] * 0.3

        save_path = output_path_loca / f"{positives[k]}.png"
        show_result(valid_composited.cpu().numpy(), coord_final,
                    img_ann[positives[k]]['bboxes'], save_path)
    return acc_num

def lerf_localization_post(sem_map, image, clip_model, image_name, img_ann,which_loc_func,is_sky_filter = False):
    '''
    Args:

    '''
    output_path_loca = image_name / 'localization'
    output_path_loca.mkdir(exist_ok=True, parents=True)
    valid_map = clip_model.get_max_across_post(sem_map, is_sky_filter)  # 3xkx832x1264
    # if  is_sky_filter:
    #     sky_valid_map = valid_map[:,:,-1,...] #[4,3,528,390]
    #     valid_map = valid_map[:,:,:-1,...] #[4,3,2,528,390]
    #     sky_valid_map = sky_valid_map.view(-1, sky_valid_map.shape[-2], sky_valid_map.shape[-1]) #[12,w,h]
    #     sky_valid_map = torch.max(sky_valid_map, dim=0)[0] #[w,h]
    #     output_sky = sky_valid_map
    #     output_sky = output_sky - torch.min(output_sky)
    #     output_sky = output_sky / (torch.max(output_sky) + 1e-9)
    #     output_sky = output_sky * (1.0 - (-1.0)) + (-1.0)
    #     output_sky = torch.clip(output_sky, 0, 1)
    #     not_sky_thresh = 0.01
    #     mask_not_sky_pred = (output_sky <not_sky_thresh).to(torch.bool)
    #     import torchvision
    #     save_vis_mask = mask_not_sky_pred[None,...].repeat(3,1,1).to(torch.float32)
    #     torchvision.utils.save_image(save_vis_mask,'/home/wangyz/Downloads/{}.jpg'.format(time.time()))
    n_var, n_head, n_prompt, h, w = valid_map.shape
    valid_map = valid_map.view(n_head * n_var, n_prompt,h,w) #[12,2,w,h]
    # positive prompts
    acc_num = 0
    keys = [key for key in img_ann.keys() if key not in ['h', 'w']]
    positives = list(keys)
    for k in range(len(positives)):
        select_output = valid_map[:, k] # [12,w,h]
        # if is_sky_filter:
        #     select_output = select_output * (mask_not_sky_pred[None,...])
        # NOTE 平滑后的激活值图中找最大值点
        scale = 30
        kernel = np.ones((scale, scale)) / (scale ** 2)
        np_relev = select_output.cpu().numpy() #[12,w,h]
        avg_filtered = cv2.filter2D(np_relev.transpose(1, 2, 0), -1, kernel)# [w,h,12]
        if which_loc_func == 'Max':
            score_lvl = np.zeros((n_head* n_var,)) # 12
            coord_lvl = []
            for i in range(n_head*n_var):
                score = avg_filtered[..., i].max()
                coord = np.nonzero(avg_filtered[..., i] == score)
                score_lvl[i] = score
                coord_lvl.append(np.asarray(coord).transpose(1, 0)[..., ::-1])
            selec_head = np.argmax(score_lvl)
            coord_final = coord_lvl[selec_head]
        elif which_loc_func == 'WeightAvg':
            score_lvl = np.zeros((n_head * n_var,)) #list 12[]
            coord_lvl = []
            for i in range(n_head * n_var):
                score = avg_filtered[..., i].max() # float
                coord = np.nonzero(avg_filtered[..., i] == score) #[X,Y]
                score_lvl[i] = score
                coord_lvl.append(np.asarray(coord).transpose(1, 0)[..., ::-1])
            from scipy.stats import gaussian_kde
            import matplotlib.pyplot as plt
            coords = np.array([c[0] for c in coord_lvl])
            scores = np.array(score_lvl)
            # 基于打分调整权重
            weights = scores / scores.sum()
            # 核密度估计
            kde = gaussian_kde(coords.T, weights=weights)
            # 可视化密度分布
            x = np.linspace(coords[:, 0].min(), coords[:, 0].max(), 100)
            y = np.linspace(coords[:, 1].min(), coords[:, 1].max(), 100)
            xx, yy = np.meshgrid(x, y)
            density = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

            plt.imshow(density, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower')
            # plt.scatter(coords[:, 0], coords[:, 1], c=scores, cmap='Reds', edgecolor='k')
            # plt.colorbar(label='Density')
            # plt.title("Density Map with Scores")
            # plt.show()
            # plt.savefig('/home/wangyz/Downloads/density.png', dpi=300)
            density = torch.tensor(density)
            max_density = torch.max(density)
            max_index = torch.argmax(density)
            max_index_2d = np.unravel_index(max_index.item(), density.shape)

            coord_final = np.array([[int(x[max_index_2d[0]]),int(y[max_index_2d[1]])]])

            for i in range(n_head*n_var):
                score = avg_filtered[..., i].max()
                coord = np.nonzero(avg_filtered[..., i] == score)
                score_lvl[i] = score
                coord_lvl.append(np.asarray(coord).transpose(1, 0)[..., ::-1])
            selec_head = np.argmax(score_lvl) # JUST USE TO VIS

        else:
            raise NotImplementedError


        for box in img_ann[positives[k]]['bboxes'].reshape(-1, 4):
            flag = 0
            x1, y1, x2, y2 = box
            x_min, x_max = min(x1, x2), max(x1, x2)
            y_min, y_max = min(y1, y2), max(y1, y2)
            for cord_list in coord_final:
                if (cord_list[0] >= x_min and cord_list[0] <= x_max and
                        cord_list[1] >= y_min and cord_list[1] <= y_max):
                    acc_num += 1
                    flag = 1
                    break
            if flag != 0:
                break

        # NOTE 将平均后的结果与原结果相加，抑制噪声并保持激活边界清晰
        avg_filtered = torch.from_numpy(avg_filtered[..., selec_head]).unsqueeze(-1).to(select_output.device)
        torch_relev = 0.5 * (avg_filtered + select_output[selec_head].unsqueeze(-1))
        p_i = torch.clip(torch_relev - 0.5, 0, 1)
        valid_composited = colormaps.apply_colormap(p_i / (p_i.max() + 1e-6), colormaps.ColormapOptions("turbo"))
        mask = (torch_relev < 0.5).squeeze()
        image = torch.Tensor(cv2.resize(image.cpu().numpy(), (mask.shape[1], mask.shape[0]))).cuda()
        valid_composited[mask, :] = image[mask, :] * 0.3

        save_path = output_path_loca / f"{positives[k]}.png"
        show_result(valid_composited.cpu().numpy(), coord_final,
                    img_ann[positives[k]]['bboxes'], save_path)
    return acc_num

def evaluate(feat_dir, output_path, ae_ckpt_path, json_folder, mask_thresh, encoder_hidden_dims, decoder_hidden_dims,
             logger,resolution,which_feature_fusion_func,is_sky_filter):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    colormap_options = colormaps.ColormapOptions(
        colormap="turbo",
        normalize=True,
        colormap_min=-1.0,
        colormap_max=1.0,
    )

    gt_ann, image_paths = eval_gt_lerfdata(Path(json_folder), Path(output_path), resolution=resolution)
    eval_index_list = [idx for idx in list(gt_ann.keys())]  # 1,24,42,106,128,139
    print('Eval_index_list')
    print(eval_index_list)
    compressed_sem_feats = [] #expectation [N_img(6),N_lvl(3),H(730),W(988),C(3)]
    for i, idx in enumerate(eval_index_list):
        compressed_sem_feats_dir = []

        for j in range(len(feat_dir)):
            feat_paths_lvl = sorted(glob.glob(os.path.join(feat_dir[j], '*.npy')),
                                    key=lambda file_name: int(os.path.basename(file_name).split(".npy")[0]))
            feat_paths_lvl = {
                os.path.basename(path).split('/')[-1][:-4]: path
                for path in feat_paths_lvl
            }

            a = np.load(feat_paths_lvl[idx]) # in default [W,H,3]; in avg [W,H,12]
            compressed_sem_feats_dir.append(a)
        compressed_sem_feats.append(compressed_sem_feats_dir)


    # instantiate autoencoder and openclip
    clip_model = OpenCLIPNetwork(device)
    checkpoint = torch.load(ae_ckpt_path, map_location=device)
    model = Autoencoder(encoder_hidden_dims, decoder_hidden_dims).to(device)
    model.load_state_dict(checkpoint)
    model.eval()

    chosen_iou_all, chosen_lvl_list = [], []
    acc_num = 0
    for j, idx in enumerate(tqdm(eval_index_list)):
        image_name = Path(output_path) / idx
        image_name.mkdir(exist_ok=True, parents=True)
        sem_feat = compressed_sem_feats[j] #list3 [W,H,C]
        # sem_feat = compressed_sem_feats[:, j, ...]
        sem_feat = torch.from_numpy(np.array(sem_feat)).float().to(device) #[3,w,h,3]

        rgb_img = cv2.imread(image_path[j])[..., ::-1] #[W,H,3]
        new_width = rgb_img.shape[1] // resolution
        new_height = rgb_img.shape[0] // resolution
        # 使用cv2的resize函数
        rgb_img = cv2.resize(rgb_img, (new_width, new_height))
        rgb_img = (rgb_img / 255.0).astype(np.float32)
        rgb_img = torch.from_numpy(rgb_img).to(device)
        with (torch.no_grad()):
            lvl = sem_feat.shape[0]
            # h = gt_ann[f'{idx}']['h']
            # w = gt_ann[f'{idx}']['w']
            # lvl, h, w, _ = sem_feat.shape
            h = sem_feat.shape[1]
            w = sem_feat.shape[2]
            if which_feature_fusion_func == 'default':
                # sem_feat = sem_feat[...,9:12] # remember to del only ablation test
                restored_feat = model.decode(sem_feat.flatten(0, 2)) # 3,....，3
                restored_feat = restored_feat.view(lvl, h, w, -1)  # 3x832x1264x512
            else:
                num_view_rendering = sem_feat.shape[3]//3 - 1
                assert num_view_rendering >= 1, 'num_view_rendering should be greater than 1'
                split_feat_list = torch.split(sem_feat, 3, dim=-1) #(tuple:num_view_rendering)
                restored_feat_list = []
                for i in range(len(split_feat_list)):
                    restored_feat = model.decode(split_feat_list[i].flatten(0, 2))
                    restored_feat = restored_feat.view(lvl, h, w, -1)  # 3x832x1264x512
                    restored_feat_list.append(restored_feat)
                restored_feat = torch.stack(restored_feat_list, dim=0) # 直接在DECODER后的dense clip 上操作
                #restored_feat :# 4x 3x832x1264x512
                if  which_feature_fusion_func == 'avg':
                    restored_feat = restored_feat.mean(dim=0)
                else:
                    raise NotImplementedError
        img_ann = gt_ann[f'{idx}']
        keys = [key for key in img_ann.keys() if key not in ['h', 'w']]

        clip_model.set_positives(list(keys))

        c_iou_list, c_lvl = activate_stream(restored_feat, rgb_img, clip_model, image_name, img_ann,
                                            thresh=mask_thresh, colormap_options=colormap_options,resolution=resolution)
        chosen_iou_all.extend(c_iou_list)
        chosen_lvl_list.extend(c_lvl)
        logger.info(f'ID:{idx},  iou chosen {c_iou_list}')
        acc_num_img = lerf_localization(restored_feat, rgb_img, clip_model, image_name, img_ann)
        acc_num += acc_num_img

    # # iou chosen
    mean_iou_chosen = sum(chosen_iou_all) / len(chosen_iou_all)
    logger.info(f'trunc thresh: {mask_thresh}')
    logger.info(f"iou chosen: {mean_iou_chosen:.4f}")
    logger.info(f"chosen_lvl: \n{chosen_lvl_list}")

    # localization acc
    total_bboxes = 0
    for img_ann in gt_ann.values():
        total_bboxes += len(list(img_ann.keys()))
    acc = acc_num / total_bboxes
    logger.info("Localization accuracy: " + f'{acc:.4f}')

def evaluate_postfusion(feat_dir, output_path, ae_ckpt_path, json_folder, mask_thresh, encoder_hidden_dims, decoder_hidden_dims,
             logger,resolution,which_feature_fusion_func,is_sky_filter):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    colormap_options = colormaps.ColormapOptions(
        colormap="turbo",
        normalize=True,
        colormap_min=-1.0,
        colormap_max=1.0,
    )

    gt_ann, image_paths = eval_gt_lerfdata(Path(json_folder), Path(output_path), resolution=resolution)
    eval_index_list = [idx for idx in list(gt_ann.keys())]  # 1,24,42,106,128,139
    print('Eval_index_list')
    print(eval_index_list)
    compressed_sem_feats = [] #expectation [N_img(6),N_lvl(3),H(730),W(988),C(3)]
    for i, idx in enumerate(eval_index_list):
        compressed_sem_feats_dir = []
        print(feat_dir)
        for j in range(len(feat_dir)):
            DEBUG = True  # Set to False to use original integer sorting fASLE is the right pt DATASET
            try:
                feat_paths_lvl = sorted(glob.glob(os.path.join(feat_dir[j], '*.npy')),
                                        key=lambda file_name: int(os.path.basename(file_name).split(".npy")[0]))
            except:
                print('not pt, attention bugs') # FIXME : a implementation for on-the-go dataset
                feat_paths_lvl = sorted(glob.glob(os.path.join(feat_dir[j], '*.npy')))

            feat_paths_lvl = {
                os.path.basename(path).split('/')[-1][:-4]: path
                for path in feat_paths_lvl
            }

            a = np.load(feat_paths_lvl[idx]) # in default [W,H,3]; in avg [W,H,12]
            compressed_sem_feats_dir.append(a)
        compressed_sem_feats.append(compressed_sem_feats_dir)

    # instantiate autoencoder and openclip
    clip_model = OpenCLIPNetwork(device)
    checkpoint = torch.load(ae_ckpt_path, map_location=device)
    model = Autoencoder(encoder_hidden_dims, decoder_hidden_dims).to(device)
    model.load_state_dict(checkpoint)
    model.eval()

    chosen_iou_all, chosen_lvl_list = [], []
    acc_num = 0
    for j, idx in enumerate(tqdm(eval_index_list)):
        image_name = Path(output_path) / idx
        print('Evaling: ',image_name)
        image_name.mkdir(exist_ok=True, parents=True)
        sem_feat = compressed_sem_feats[j] #list3 [W,H,C]
        # sem_feat = compressed_sem_feats[:, j, ...]
        sem_feat = torch.from_numpy(np.array(sem_feat)).float().to(device) #[3,w,h,3]
        rgb_img = cv2.imread(image_paths[j])[..., ::-1] #[W,H,3]
        new_width = rgb_img.shape[1] // resolution
        new_height = rgb_img.shape[0] // resolution
        # 使用cv2的resize函数
        rgb_img = cv2.resize(rgb_img, (new_width, new_height))
        rgb_img = (rgb_img / 255.0).astype(np.float32)
        rgb_img = torch.from_numpy(rgb_img).to(device)
        with (torch.no_grad()):
            lvl = sem_feat.shape[0]
            # h = gt_ann[f'{idx}']['h']
            # w = gt_ann[f'{idx}']['w']
            # lvl, h, w, _ = sem_feat.shape
            h = sem_feat.shape[1]
            w = sem_feat.shape[2]

            # all which_feature_fusion_func starts with 'post', must contain the virtual appearance;
            num_view_rendering = sem_feat.shape[3]//3 - 1
            assert num_view_rendering >= 1, 'num_view_rendering should be greater than 1'
            split_feat_list = torch.split(sem_feat, 3, dim=-1) #(tuple:num_view_rendering)
            restored_feat_list = []
            for i in range(len(split_feat_list)):
                    restored_feat = model.decode(split_feat_list[i].flatten(0, 2))
                    restored_feat = restored_feat.view(lvl, h, w, -1)  # 3x832x1264x512
                    restored_feat_list.append(restored_feat)
            restored_feat = torch.stack(restored_feat_list, dim=0) # 直接在DECODER后的dense clip 上操作
            #restored_feat :# 4x 3x832x1264x512
        img_ann = gt_ann[f'{idx}']
        keys = [key for key in img_ann.keys() if key not in ['h', 'w']]
        if is_sky_filter:
            keys.append("sky")
        clip_model.set_positives(list(keys))

        which_fusion_func, which_loc_func = which_feature_fusion_func.split('|Loc')

        c_iou_list, c_lvl = activate_stream_post(restored_feat, rgb_img, clip_model, image_name, img_ann,
                                            thresh=mask_thresh, colormap_options=colormap_options,resolution=resolution,which_feature_fusion_func = which_fusion_func,is_sky_filter = is_sky_filter)
        chosen_iou_all.extend(c_iou_list)
        chosen_lvl_list.extend(c_lvl)
        logger.info(f'ID:{idx},  iou chosen {c_iou_list}')
        acc_num_img = lerf_localization_post(restored_feat, rgb_img, clip_model, image_name, img_ann,which_loc_func, is_sky_filter = is_sky_filter)
        acc_num += acc_num_img

    # # iou chosen
    mean_iou_chosen = sum(chosen_iou_all) / len(chosen_iou_all)
    logger.info(f'trunc thresh: {mask_thresh}')
    logger.info(f"iou chosen: {mean_iou_chosen:.4f}")
    logger.info(f"chosen_lvl: \n{chosen_lvl_list}")

    # localization acc
    total_bboxes = 0
    for img_ann in gt_ann.values():
        total_bboxes += len(list(img_ann.keys()))
    acc = acc_num / total_bboxes
    logger.info("Localization accuracy: " + f'{acc:.4f}')


def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value )

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


if __name__ == "__main__":
    seed_num = 42
    seed_everything(seed_num)
    print('Evaluating the results......')
    parser = ArgumentParser(description="prompt any label")
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument('--feat_dir', type=str, default=None)
    parser.add_argument("--ae_ckpt_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--json_folder", type=str, default=None)
    parser.add_argument("--mask_thresh", type=float, default=0.4)
    parser.add_argument("--resolution", type=float, default=2)
    parser.add_argument('--encoder_dims',
                        nargs='+',
                        type=int,
                        default=[256, 128, 64, 32, 3],
                        )
    parser.add_argument('--decoder_dims',
                        nargs='+',
                        type=int,
                        default=[16, 32, 64, 128, 256, 256, 512],
                        )
    parser.add_argument('--which_feature_fusion_func',type=str,default='post_validMapLevel_avgImageWiseMaxValue|LocMax', \
                        choices=['default','avg','post_validMapLevel_avg|LocMax', 'post_validMapLevel_max|LocMax','post_resultLevel_globalMax|LocMax',\
                                 'post_validMapLevel_avgPixelwiseNormalDistribution|LocMax','post_validMapLevel_avgImageWiseNormalDistribution|LocMax','post_validMapLevel_avgImageWiseMaxValue|LocMax',\
                                 'post_validMapLevel_avgHybridImageWiseMaxValuePixelWiseMaxValue|LocMax','post_validMapLevel_avgImageWiseMaxValue|LocMax','post_validMapLevel_avgImageWiseMaxValue|LocWeightAvg'])
    parser.add_argument('--sky_filter', action='store_true')
    args = parser.parse_args()

    # NOTE config setting
    dataset_name = args.dataset_name
    mask_thresh = args.mask_thresh
    feat_dir = [os.path.join(args.feat_dir, dataset_name + f"_{i}", "train/ours_None/renders_npy") for i in range(1, 4)]
    # feat_dir = [os.path.join(args.feat_dir, dataset_name + f"_{i}", "test/ours_None/renders_npy") for i in range(1, 4)]
    output_path = os.path.join(args.output_dir, dataset_name+'['+args.which_feature_fusion_func+']')
    # dataset_name:wegs_vanillaCLIPExt_VanillaRecon-?dataset_name.split('_')[1]:vanillaCLIPExt
    ae_ckpt_path = os.path.join(args.ae_ckpt_dir, dataset_name, "best_ckpt.pth")
    print('ae_ckpt:',ae_ckpt_path)
    # ae_ckpt_path = '/home/wangyz/Documents/projects/0working/langsplat-w/autoencoder/ckpt/VanillaCLIPExt-100epoch/best_ckpt.pth' #dirty code for debug
    json_folder = os.path.join(args.json_folder)

    # NOTE logger
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    os.makedirs(output_path, exist_ok=True)
    log_file = os.path.join(output_path, f'{timestamp}.log')
    logger = get_logger(f'{dataset_name}', log_file=log_file, log_level=logging.INFO)
    if args.which_feature_fusion_func.startswith('post'):
        evaluate_postfusion(feat_dir, output_path, ae_ckpt_path, json_folder, mask_thresh, args.encoder_dims, args.decoder_dims,logger,args.resolution,args.which_feature_fusion_func,args.sky_filter)
    else:
        evaluate(feat_dir, output_path, ae_ckpt_path, json_folder, mask_thresh, args.encoder_dims, args.decoder_dims,logger,args.resolution,args.which_feature_fusion_func, args.sky_filter)
