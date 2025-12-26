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
from tqdm import tqdm

import sys

sys.path.append("..")
import colormaps
from autoencoder.model import Autoencoder
from openclip_encoder import OpenCLIPNetwork
from utils import smooth, colormap_saving, vis_mask_save, polygon_to_mask, stack_mask, show_result



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



def evaluate(rgb_path_list,img_list,feature_list,text_prompt, output_path,gt_json_folder, mask_thresh ,resolution,logger):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gt_json_paths = sorted(glob.glob(os.path.join(str(gt_json_folder), '*.json')))
    img_paths = sorted(glob.glob(os.path.join(str(gt_json_folder), '*.jpg')))
    gt_ann = {}
    for js_path in gt_json_paths:
        img_ann = defaultdict(dict)
        with open(js_path, 'r') as f:
            gt_data = json.load(f)
        h, w = gt_data['info']['height'], gt_data['info']['width']
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
            save_path = Path(output_path) / 'gt' / gt_data['info']['name'].split('.jpg')[0] / f'{label}.jpg'
            save_path.parent.mkdir(exist_ok=True, parents=True)
            vis_mask_save(mask, save_path)
        gt_ann[f'{idx}'] = img_ann
    colormap_options = colormaps.ColormapOptions(
        colormap="turbo",
        normalize=False,
        colormap_min=-1.0,
        colormap_max=1.0,
    )
    clip_model = OpenCLIPNetwork(device)
    clip_model.set_positives(text_prompt)
    # positive prompts
    chosen_iou_all = []
    chosen_lvl_all = []
    flag = 0
    for j in range(len(rgb_path_list)):
        chosen_iou_list, chosen_lvl_list = [], []
        rgb_path = rgb_path_list[j]
        image_path = img_list[j]
        img_ann = gt_ann[image_path]
        clip_feature = torch.Tensor(feature_list[j]).float().to(device)
        try:
            rgb_img = cv2.imread(rgb_path)[..., ::-1]
        except:
            print(rgb_path)
            exit(-1)
        valid_map = clip_model.get_max_across(clip_feature) #[lvl, text_prompt, img_h, img_w]
        n_head, n_prompt, h, w = valid_map.shape
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

                output_path_relev = Path(output_path) /image_path/ 'heatmap' / f'{clip_model.positives[k]}_{i}'
                output_path_relev.parent.mkdir(exist_ok=True, parents=True)
                colormap_saving(valid_map[i][k].unsqueeze(-1), colormap_options,
                                output_path_relev)

                # NOTE 与lerf一致，激活值低于0.5的认为是背景
                p_i = torch.clip(valid_map[i][k] - 0.5, 0, 1).unsqueeze(-1)
                valid_composited = colormaps.apply_colormap(p_i / (p_i.max() + 1e-6),
                                                            colormaps.ColormapOptions("turbo"))
                mask = (valid_map[i][k] < 0.5).squeeze()
                image = torch.Tensor(cv2.resize(rgb_img/255.0, (mask.shape[1], mask.shape[0]))).cuda()
                valid_composited[mask, :] = image[mask, :] * 0.3
                output_path_compo = Path(output_path) /image_path/ 'composited' / f'{clip_model.positives[k]}_{i}'
                output_path_compo.parent.mkdir(exist_ok=True, parents=True)
                colormap_saving(valid_composited, colormap_options, output_path_compo)

                output = valid_map[i][k]
                output = output - torch.min(output)
                output = output / (torch.max(output) + 1e-9)
                output = output * (1.0 - (-1.0)) + (-1.0)
                output = torch.clip(output, 0, 1)

                mask_pred = (output.cpu().numpy() > mask_thresh).astype(np.uint8)
                mask_pred = smooth(mask_pred)
                mask_lvl[i] = mask_pred
                try:
                    mask_gt = img_ann[clip_model.positives[k]]['mask'].astype(np.uint8)
                except:
                    # print(
                    #     f'warning: {clip_model.positives[k]} does not have mask in {img_ann[clip_model.positives[k]]}')
                    # mask_gt = np.zeros_like(mask_pred)
                    flag = 1
                    continue
                mask_gt = cv2.resize(mask_gt, (mask_pred.shape[1], mask_pred.shape[0]))
                # calculate iou
                intersection = np.sum(np.logical_and(mask_gt, mask_pred))
                union = np.sum(np.logical_or(mask_gt, mask_pred))
                iou = np.sum(intersection) / np.sum(union)
                iou_lvl[i] = iou
            if flag != 1:
                score_lvl = torch.zeros((n_head,), device=valid_map.device)
                for i in range(n_head):
                    score = valid_map[i, k].max()
                    score_lvl[i] = score
                chosen_lvl = torch.argmax(score_lvl)

                chosen_iou_list.append(iou_lvl[chosen_lvl])
                chosen_lvl_list.append(chosen_lvl.cpu().numpy())
                # save for visulsization
                save_path = Path(output_path)/image_path / f'chosen_{clip_model.positives[k]}.png'
                vis_mask_save(mask_lvl[chosen_lvl], save_path)
            else:
                flag = 0
        chosen_iou_all.extend(chosen_iou_list)
        chosen_lvl_list.extend(chosen_lvl_list)
    # # iou
    # mean_iou_chosen = sum(chosen_iou_all) / len(chosen_iou_all)
    # logger.info(f'trunc thresh: {mask_thresh}')
    # logger.info(f"iou chosen: {mean_iou_chosen:.4f}")
    # logger.info(f"chosen_lvl: \n{chosen_lvl_list}")


def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

def generate_clip_feature(f_path_list,s_path_list):
    '''
    output:[3,w,h,512]
    '''
    feature_list = []
    for i in range(len(f_path_list)):
        f = np.load(f_path_list[i])
        s = np.load(s_path_list[i]).astype(int)
        feature = f[s][1:]
        feature_list.append(feature)
    return feature_list
if __name__ == "__main__":
    seed_num = 42
    seed_everything(seed_num)

    parser = ArgumentParser(description="prompt any label")
    parser.add_argument("--source_dataset",type=str,default="/media/wangyz/DATA/UBUNTU_data/dataset/PT/notre_dame_front_facade")
    parser.add_argument("--img_list", default=['95329622_1056099621','96866778_4045929561','97540424_7796073188','97624324_6389845793','98084027_3183279350','98182807_6864978761',\
                                               '98341766_2349157681','98368222_6582890053','98773057_1661405515','98912687_6099556076','99051180_197217963','99442465_5374619776',\
                                               '99592053_2251819227','99652714_3819900868','99784014_3777253105','99837444_10018180103','99856798_307226100','99895792_5135186752'])
    parser.add_argument('--text_prompt_list_folder', default='/media/wangyz/DATA/UBUNTU_data/dataset/PT/text_prompt_list.txt')
    parser.add_argument('--gt_json_folder', default='/media/wangyz/DATA/UBUNTU_data/dataset/PT/label/notre_dame_front_facade')
    parser.add_argument("--output_folder", type=str, default='/home/wangyz/Downloads/notre_dame_front_facade')
    parser.add_argument("--mask_thresh", type=float, default=0.4)
    parser.add_argument("--resolution", type=float, default=2)


    # parser.add_argument("--source_dataset",type=str,default="/media/wangyz/DATA/UBUNTU_data/dataset/lerf_ovs/teatime")
    # parser.add_argument("--img_list", default=['frame_00002', 'frame_00025', 'frame_00043','frame_00107','frame_00129','frame_00140'])
    # parser.add_argument('--text_prompt_list_folder',
    #                     default='/media/wangyz/DATA/UBUNTU_data/dataset/lerf_ovs/label/text_prompt_list_for_teatime.txt')
    # parser.add_argument('--gt_json_folder',
    #                     default='/media/wangyz/DATA/UBUNTU_data/dataset/lerf_ovs/label/teatime')
    # parser.add_argument("--output_folder", type=str, default='/home/wangyz/Downloads/test_teatime')
    # parser.add_argument("--mask_thresh", type=float, default=0.4)
    # parser.add_argument("--resolution", type=float, default=1)
    args = parser.parse_args()

    # NOTE config setting
    source_dataset = args.source_dataset
    mask_thresh = args.mask_thresh
    # rgb_path_list = [os.path.join(source_dataset, "images", img_name + '.jpg') for img_name in args.img_list]
    rgb_path_list = [os.path.join(source_dataset, "dense", "images",img_name+'.jpg') for img_name in args.img_list]
    f_path_list = [os.path.join(source_dataset, "language_features",img_name+"_f.npy") for img_name in args.img_list]
    s_path_list = [os.path.join(source_dataset, "language_features", img_name + "_s.npy") for img_name in args.img_list]
    output_path = args.output_folder
    gt_json_folder = args.gt_json_folder
    os.makedirs(output_path, exist_ok=True)
    # 初始化一个空列表
    text_prompt_list = []
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(output_path, f'{timestamp}.log')
    logger = get_logger(f'test', log_file=log_file, log_level=logging.INFO)

    # 读取文件内容并按行存储到列表中
    with open(args.text_prompt_list_folder, 'r', encoding='utf-8') as file:
        text_prompt_list = file.read().splitlines()

    feature_list = generate_clip_feature(f_path_list,s_path_list)
    evaluate(rgb_path_list,args.img_list,feature_list,text_prompt_list, output_path, gt_json_folder, mask_thresh ,args.resolution,logger)
