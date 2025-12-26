import os
import random
import argparse
# hello? 20240923
import numpy as np
import torch
from skimage.util import img_as_bool
from sympy.physics.units import action
from sympy.polys.polyconfig import query
from tqdm import tqdm
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import cv2

from dataclasses import dataclass, field
from typing import Tuple, Type
from copy import deepcopy

import torch
import torchvision
from torch import nn

from arguments import itw_ModelParams,itw_PipelineParams,itw_AEParams,itw_TEParams
from in_the_wild_renderer import WEGSRenderer
try:
    import open_clip
except ImportError:
    assert False, "open_clip is not installed, install it with `pip install open-clip-torch`"


@dataclass
class OpenCLIPNetworkConfig:
    _target: Type = field(default_factory=lambda: OpenCLIPNetwork)
    clip_model_type: str = "ViT-B-16"
    clip_model_pretrained: str = "laion2b_s34b_b88k"
    clip_n_dims: int = 512
    negatives: Tuple[str] = ("object", "things", "stuff", "texture")
    positives: Tuple[str] = ("",)


class OpenCLIPNetwork(nn.Module):
    def __init__(self, config: OpenCLIPNetworkConfig):
        super().__init__()
        self.config = config
        self.process = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )
        model, _, _ = open_clip.create_model_and_transforms(
            self.config.clip_model_type,  # e.g., ViT-B-16
            pretrained=self.config.clip_model_pretrained,  # e.g., laion2b_s34b_b88k
            precision="fp16",
        )
        model.eval()
        self.tokenizer = open_clip.get_tokenizer(self.config.clip_model_type)
        self.model = model.to("cuda")
        self.clip_n_dims = self.config.clip_n_dims

        self.positives = self.config.positives
        self.negatives = self.config.negatives
        with torch.no_grad():
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.positives]).to("cuda")
            self.pos_embeds = model.encode_text(tok_phrases)
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.negatives]).to("cuda")
            self.neg_embeds = model.encode_text(tok_phrases)
        self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)
        self.neg_embeds /= self.neg_embeds.norm(dim=-1, keepdim=True)

        assert (
                self.pos_embeds.shape[1] == self.neg_embeds.shape[1]
        ), "Positive and negative embeddings must have the same dimensionality"
        assert (
                self.pos_embeds.shape[1] == self.clip_n_dims
        ), "Embedding dimensionality must match the model dimensionality"

    @property
    def name(self) -> str:
        return "openclip_{}_{}".format(self.config.clip_model_type, self.config.clip_model_pretrained)

    @property
    def embedding_dim(self) -> int:
        return self.config.clip_n_dims

    def gui_cb(self, element):
        self.set_positives(element.value.split(";"))

    def set_positives(self, text_list):
        self.positives = text_list
        with torch.no_grad():
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.positives]).to("cuda")
            self.pos_embeds = self.model.encode_text(tok_phrases)
        self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)

    def get_relevancy(self, embed: torch.Tensor, positive_id: int) -> torch.Tensor:
        phrases_embeds = torch.cat([self.pos_embeds, self.neg_embeds], dim=0)
        p = phrases_embeds.to(embed.dtype)  # phrases x 512
        output = torch.mm(embed, p.T)  # rays x phrases
        positive_vals = output[..., positive_id: positive_id + 1]  # rays x 1
        negative_vals = output[..., len(self.positives):]  # rays x N_phrase
        repeated_pos = positive_vals.repeat(1, len(self.negatives))  # rays x N_phrase

        sims = torch.stack((repeated_pos, negative_vals), dim=-1)  # rays x N-phrase x 2
        softmax = torch.softmax(10 * sims, dim=-1)  # rays x n-phrase x 2
        best_id = softmax[..., 0].argmin(dim=1)  # rays x 2
        return torch.gather(softmax, 1, best_id[..., None, None].expand(best_id.shape[0], len(self.negatives), 2))[:, 0,
               :]

    def encode_image(self, input):
        processed_input = self.process(input).half()
        return self.model.encode_image(processed_input)



def predict_pixel_level_clip_from_image(img, sam_encoder,color_palette = None, debug = True):
    img_embed, seg_map = _embed_clip_sam_tiles(img, sam_encoder)
    lengths = [len(v) for k, v in img_embed.items()]
    total_length = sum(lengths)
    # total_lengths.append(total_length)
    img_embed = torch.cat([v for k, v in img_embed.items()], dim=0)
    seg_map_tensor = []
    seg_vis_map_tensor = []
    seg_vis_map = None
    lengths_cumsum = lengths.copy()
    for j in range(1, len(lengths)):
        lengths_cumsum[j] += lengths_cumsum[j - 1]
    for j, (k, v) in enumerate(seg_map.items()):
        if j == 0:
            seg_map_tensor.append(torch.from_numpy(v))
            if debug:
                seg_vis_map_tensor.append(torch.from_numpy(color_palette[v]).permute(2, 0, 1).float() / 255.0)
            continue
        assert v.max() == lengths[j] - 1, f"{j}, {v.max()}, {lengths[j] - 1}"
        v[v != -1] += lengths_cumsum[j - 1]
        seg_map_tensor.append(torch.from_numpy(v))
        if debug:
            seg_vis_map_tensor.append(torch.from_numpy(color_palette[v]).permute(2, 0, 1).float() / 255.0)
    seg_map = torch.stack(seg_map_tensor, dim=0) #[4,w,h]
    if debug:
        seg_vis_map = torch.stack(seg_vis_map_tensor, dim=0)

    return img_embed, seg_map, seg_vis_map, total_length # img_embed:[N_type_across_scale,512],seg_map:[4,w,h],seg_vis_map[4,3,w,h]

def create(image_list, img_name_list, save_folder,aug_ma_renderings, debug=True):


    assert image_list is not None, "image_list must be provided to generate features"
    embed_size = 512
    n_aug_ma = next(iter(aug_ma_renderings.values())).shape[0]
    total_lengths = []
    img_embeds = torch.zeros((len(image_list),n_aug_ma + 1, 300, embed_size)) # [N_imgs, origin_View + N_render_view, Length_of_seg, 512]
    color_palette = None
    if debug:
        color_palette = np.random.randint(0, 256, (65536, 3), dtype=np.uint8)

    seg_maps = []
    seg_vis_maps = []

    tm_list = []
    am_list = []
    tm_min, tm_max = torch.inf,0
    am_min, am_max = torch.inf,0
    m_path_list = []
    mask_generator.predictor.model.to('cuda')
    print('Total Image:',len(image_list))

    for i, img in tqdm(enumerate(image_list), desc="Embedding images", leave=False):

        # try:
            img_name = img_name_list[i].split('.')[0]
            aug_ma_rendering = aug_ma_renderings[img_name]
            img_embed_list = []
            seg_map_list = []
            seg_vis_map_list = []
            total_length_list = []
            img_embed_origin, seg_map_origin, seg_vis_map_origin, total_length_origin = predict_pixel_level_clip_from_image(
                img, sam_encoder, color_palette, debug)
            img_embed_list.append(img_embed_origin)
            seg_map_list.append(seg_map_origin)
            seg_vis_map_list.append(seg_vis_map_origin)
            total_length_list.append(total_length_origin)
            for j , aug_img in enumerate(aug_ma_rendering):
                aug_img = torch.from_numpy((aug_img * 255).astype(np.uint8))[None,...]
                img_embed_aug, seg_map_aug, seg_vis_map_aug, total_length_aug = predict_pixel_level_clip_from_image(aug_img, sam_encoder, color_palette, debug)
                img_embed_list.append(img_embed_aug)
                seg_map_list.append(seg_map_aug)
                seg_vis_map_list.append(seg_vis_map_aug)
                total_length_list.append(total_length_aug)
            total_lengths.append(total_length_list)
            # write somewhat a little dirty from the vanilla LangSplat. extent the Arr. img_embeds to the total_length across every images
            for total_length in total_length_list:
                if total_length > img_embeds.shape[2]:
                    pad = total_length - img_embeds.shape[2]
                    img_embeds = torch.cat([
                        img_embeds,
                        torch.zeros((len(image_list),n_aug_ma+1, pad, embed_size))
                    ], dim=2)
            for ith_na, img_embed in enumerate(img_embed_list):
                img_embeds[i,ith_na,:total_length_list[ith_na]] =img_embed  # i-th, 0-total_length-1 mask,else 0
            seg_maps.append(seg_map_list)#List N_num of [4,w,h]
            seg_vis_maps.append(seg_vis_map_list)  #List: N_num of [4,3,w,h]
            # save imgs
            save_path = os.path.join(save_folder, img_name_list[i].split('.')[0])
            m_path_list.append(save_path)
            curr = {}
            feature_aug_list = []
            # write somewhat a little dirty from the vanilla LangSplat.-

            for idx_aug_ma in range(n_aug_ma + 1):
                if idx_aug_ma == 0:
                    curr[f'feature'] = img_embeds[i,0,:total_lengths[i][0]]
                    curr[f'seg_maps'] = seg_maps[i][0]
                    origin_feature = img_embeds[i, 0, :total_lengths[i][0]][seg_maps[i][0]]
                else:
                    curr[f'feature_ma{idx_aug_ma-1}'] = img_embeds[i, idx_aug_ma, :total_lengths[i][idx_aug_ma]]
                    curr[f'seg_maps_ma{idx_aug_ma-1}'] = seg_maps[i][idx_aug_ma]
                    aug_feature = img_embeds[i, idx_aug_ma, :total_lengths[i][idx_aug_ma]][seg_maps[i][idx_aug_ma]]
                    feature_aug_list.append(aug_feature)

            feature_aug_cat_self_appearance = torch.from_numpy(np.array(feature_aug_list[-1]))  # [3,W,H,512]]
            # clip_feature_fused = torch.mean(feature_aug_cat, dim=0) #[3,W,H,512]
            # origin_feature, clip_feature_fused # [3,W,H,512]
            diff_agg = torch.mean(torch.square(feature_aug_cat_self_appearance - origin_feature), dim=3)# [3,W,H]
            uncertainly_map_transient = torch.mean(diff_agg, dim=0)
            uncertainly_map_transient = uncertainly_map_transient.cpu().numpy()
            tm_list.append(uncertainly_map_transient)
            tm_min = min(uncertainly_map_transient.min(),tm_min)
            tm_max = max(uncertainly_map_transient.max(), tm_max)

            feature_aug_cat_virtual_appearance = torch.from_numpy(np.array(feature_aug_list[:-1]))  # [3,3,W,H,512]
            mean_feature_aug_cat_virtual_appearance = torch.mean(feature_aug_cat_virtual_appearance,dim=0)[None,...]
            uncertainly_map_appearance = torch.mean(torch.mean(torch.mean(torch.square(feature_aug_cat_virtual_appearance - mean_feature_aug_cat_virtual_appearance),dim=-1),dim=0),dim=0)
            uncertainly_map_appearance = uncertainly_map_appearance.cpu().numpy()
            am_list.append(uncertainly_map_appearance)
            am_min = min(uncertainly_map_appearance.min(),am_min)
            am_max = max(uncertainly_map_appearance.max(), am_max)
            # uncertainly_map_appearance = (uncertainly_map_appearance - uncertainly_map_appearance.min()) / (
            #             uncertainly_map_appearance.max() - uncertainly_map_appearance.min())

            save_numpy(save_path, curr)
        # except:
        #     print('ERROR-----')
        #     print(i)
        #     print(img_name_list[i])
        #     print('ERROR-----')
        #     exit(-1)
    for i in range(len(tm_list)):
        tm = (tm_list[i] - tm_min) / (tm_max - tm_min)
        am = (am_list[i] - am_min) / (am_max - am_min)
        save_mask(m_path_list[i],tm,'T')
        save_mask(m_path_list[i], am, 'A')


def save_mask(save_path,uncertain_map,flag_name):
    #flag_name = 'T' or 'A'
    save_path_um = save_path + '_uncertainly_map_{}.npy'.format(flag_name)
    np.save(save_path_um, uncertain_map)
    uncertainly_map = torch.from_numpy(uncertain_map)
    uncertainly_map = uncertainly_map[None, ...].repeat(3, 1, 1)[None, ...]
    torchvision.utils.save_image(uncertainly_map, save_path + '_uncertainly_map_{}.jpg'.format(flag_name))

# def save_vis(save_path, seg_vis_map,uncertainly_map1,uncertainly_map2):
#     uncertainly_map_t = torch.from_numpy(uncertainly_map1)
#     uncertainly_map_t = uncertainly_map_t[None,...].repeat(3,1,1)[None,...]
#     torchvision.utils.save_image(uncertainly_map_t, save_path + '_uncertainly_map_T.jpg')
#     uncertainly_map_t = torch.from_numpy(uncertainly_map2)
#     uncertainly_map_t = uncertainly_map_t[None,...].repeat(3,1,1)[None,...]
#     torchvision.utils.save_image(uncertainly_map_t, save_path + '_uncertainly_map_A.jpg')
#     for i in range(len(seg_vis_map)):
#         if i == 0:
#             save_path_default = save_path + '_vis_d.jpg'
#             save_path_small = save_path + '_vis_s.jpg'
#             save_path_middle = save_path + '_vis_m.jpg'
#             save_path_large = save_path + '_vis_l.jpg'
#         else:
#             save_path_default = save_path +'vis_d_ma_{}.jpg'.format(i)
#             save_path_small = save_path + '_vis_s_ma_{}.jpg'.format(i)
#             save_path_middle = save_path + '_vis_m_ma_{}.jpg'.format(i)
#             save_path_large = save_path + '_vis_l_ma_{}.jpg'.format(i)
#         torchvision.utils.save_image(seg_vis_map[i][0], save_path_default)
#         torchvision.utils.save_image(seg_vis_map[i][1], save_path_small)
#         torchvision.utils.save_image(seg_vis_map[i][2], save_path_middle)
#         torchvision.utils.save_image(seg_vis_map[i][3], save_path_large)


def save_numpy(save_path, data):
    save_path_s = save_path + '_s.npy'
    save_path_f = save_path + '_f.npy'
    np.save(save_path_s, data['seg_maps'].numpy())
    np.save(save_path_f, data['feature'].numpy())
    feature_ma_key_list = [k for k in data.keys() if k.startswith('feature_')]
    for key in feature_ma_key_list:
        np.save(save_path+'_f_ma_'+key[-1]+'.npy',data[key].numpy())
    seg_ma_key_list = [k for k in data.keys() if k.startswith('seg_maps_')]
    for key in seg_ma_key_list:
        np.save(save_path+'_s_ma_'+key[-1]+'.npy',data[key].numpy())



def _embed_clip_sam_tiles(image, sam_encoder):
    aug_imgs = torch.cat([image])
    seg_images, seg_map = sam_encoder(aug_imgs)

    clip_embeds = {}
    for mode in ['default', 's', 'm', 'l']:
        tiles = seg_images[mode]# [N_seg,3,224,224]
        tiles = tiles.to("cuda")
        with torch.no_grad():
            clip_embed = model.encode_image(tiles) #[N_seg,512]
        clip_embed /= clip_embed.norm(dim=-1, keepdim=True)  # [88,512],[117,512],[62,512],[33,512]
        clip_embeds[mode] = clip_embed.detach().cpu().half()

    return clip_embeds, seg_map,  # clip_embeds:dict{'scale':[N_seg,512]}     seg_map:dict{'scale':[w,h]}


def get_seg_img(mask, image):
    image = image.copy()
    image[mask['segmentation'] == 0] = np.array([0, 0, 0], dtype=np.uint8)
    x, y, w, h = np.int32(mask['bbox'])
    seg_img = image[y:y + h, x:x + w, ...]
    return seg_img


def pad_img(img):
    h, w, _ = img.shape
    l = max(w, h)
    pad = np.zeros((l, l, 3), dtype=np.uint8)
    if h > w:
        pad[:, (h - w) // 2:(h - w) // 2 + w, :] = img
    else:
        pad[(w - h) // 2:(w - h) // 2 + h, :, :] = img
    return pad


def filter(keep: torch.Tensor, masks_result) -> None:
    keep = keep.int().cpu().numpy()
    result_keep = []
    for i, m in enumerate(masks_result):
        if i in keep: result_keep.append(m)
    return result_keep



def mask_nms(masks, scores, iou_thr=0.7, score_thr=0.1, inner_thr=0.2, **kwargs):
    """
    Perform mask non-maximum suppression (NMS) on a set of masks based on their scores.

    Args:
        masks (torch.Tensor): has shape (num_masks, H, W)
        scores (torch.Tensor): The scores of the masks, has shape (num_masks,)
        iou_thr (float, optional): The threshold for IoU.
        score_thr (float, optional): The threshold for the mask scores.
        inner_thr (float, optional): The threshold for the overlap rate.
        **kwargs: Additional keyword arguments.
    Returns:
        selected_idx (torch.Tensor): A tensor representing the selected indices of the masks after NMS.
    """

    scores, idx = scores.sort(0, descending=True)
    num_masks = idx.shape[0]

    masks_ord = masks[idx.view(-1), :]
    masks_area = torch.sum(masks_ord, dim=(1, 2), dtype=torch.float)

    iou_matrix = torch.zeros((num_masks,) * 2, dtype=torch.float, device=masks.device)
    inner_iou_matrix = torch.zeros((num_masks,) * 2, dtype=torch.float, device=masks.device)
    for i in range(num_masks):
        for j in range(i, num_masks):
            intersection = torch.sum(torch.logical_and(masks_ord[i], masks_ord[j]), dtype=torch.float)
            union = torch.sum(torch.logical_or(masks_ord[i], masks_ord[j]), dtype=torch.float)
            iou = intersection / union
            iou_matrix[i, j] = iou
            # select mask pairs that may have a severe internal relationship
            if intersection / masks_area[i] < 0.5 and intersection / masks_area[j] >= 0.85:
                inner_iou = 1 - (intersection / masks_area[j]) * (intersection / masks_area[i])
                inner_iou_matrix[i, j] = inner_iou
            if intersection / masks_area[i] >= 0.85 and intersection / masks_area[j] < 0.5:
                inner_iou = 1 - (intersection / masks_area[j]) * (intersection / masks_area[i])
                inner_iou_matrix[j, i] = inner_iou

    iou_matrix.triu_(diagonal=1)
    iou_max, _ = iou_matrix.max(dim=0)
    inner_iou_matrix_u = torch.triu(inner_iou_matrix, diagonal=1)
    inner_iou_max_u, _ = inner_iou_matrix_u.max(dim=0)
    inner_iou_matrix_l = torch.tril(inner_iou_matrix, diagonal=1)
    inner_iou_max_l, _ = inner_iou_matrix_l.max(dim=0)

    keep = iou_max <= iou_thr
    keep_conf = scores > score_thr
    keep_inner_u = inner_iou_max_u <= 1 - inner_thr
    keep_inner_l = inner_iou_max_l <= 1 - inner_thr

    # If there are no masks with scores above threshold, the top 3 masks are selected
    if keep_conf.sum() == 0:
        index = scores.topk(3).indices
        keep_conf[index, 0] = True
    if keep_inner_u.sum() == 0:
        index = scores.topk(3).indices
        keep_inner_u[index, 0] = True
    if keep_inner_l.sum() == 0:
        index = scores.topk(3).indices
        keep_inner_l[index, 0] = True
    keep *= keep_conf
    keep *= keep_inner_u
    keep *= keep_inner_l

    selected_idx = idx[keep]
    return selected_idx


def masks_update(*args, **kwargs):
    # remove redundant masks based on the scores and overlap rate between masks
    masks_new = ()
    for masks_lvl in (args):
        seg_pred = torch.from_numpy(np.stack([m['segmentation'] for m in masks_lvl], axis=0))
        iou_pred = torch.from_numpy(np.stack([m['predicted_iou'] for m in masks_lvl], axis=0))
        stability = torch.from_numpy(np.stack([m['stability_score'] for m in masks_lvl], axis=0))

        scores = stability * iou_pred
        keep_mask_nms = mask_nms(seg_pred, scores, **kwargs)
        masks_lvl = filter(keep_mask_nms, masks_lvl)

        masks_new += (masks_lvl,)
    return masks_new


def sam_encoder(image):
    image = cv2.cvtColor(image[0].permute(1, 2, 0).numpy().astype(np.uint8), cv2.COLOR_BGR2RGB)
    # pre-compute masks
    masks_default, masks_s, masks_m, masks_l = mask_generator.generate(image)
    # pre-compute postprocess
    masks_default, masks_s, masks_m, masks_l = \
        masks_update(masks_default, masks_s, masks_m, masks_l, iou_thr=0.8, score_thr=0.7, inner_thr=0.5)

    def mask2segmap(masks, image):
        seg_img_list = []
        seg_map = -np.ones(image.shape[:2], dtype=np.int32)
        for i in range(len(masks)):
            mask = masks[i]
            seg_img = get_seg_img(mask, image)
            try:
                pad_seg_img = cv2.resize(pad_img(seg_img), (224, 224))
            except:
                pad_seg_img = np.zeros((224, 224, 3), dtype=np.uint8)
            seg_img_list.append(pad_seg_img)

            seg_map[masks[i]['segmentation']] = i
        if len(seg_img_list) != 0:
            seg_imgs = np.stack(seg_img_list, axis=0)  # b,H,W,3
            seg_imgs = (torch.from_numpy(seg_imgs.astype("float32")).permute(0, 3, 1, 2) / 255.0).to('cuda')

        return seg_imgs, seg_map

    seg_images, seg_maps = {}, {}
    seg_images['default'], seg_maps['default'] = mask2segmap(masks_default, image)
    if len(masks_s) != 0:
        seg_images['s'], seg_maps['s'] = mask2segmap(masks_s, image)
    if len(masks_m) != 0:
        seg_images['m'], seg_maps['m'] = mask2segmap(masks_m, image)
    if len(masks_l) != 0:
        seg_images['l'], seg_maps['l'] = mask2segmap(masks_l, image)

    # 0:default 1:s 2:m 3:l
    return seg_images, seg_maps


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


if __name__ == '__main__':
    seed_num = 42
    seed_everything(seed_num)

    parser = argparse.ArgumentParser()
    itw_model = itw_ModelParams(parser)
    itw_ap = itw_AEParams(parser)
    itw_te = itw_TEParams(parser)
    itw_pipeline = itw_PipelineParams(parser)
    parser.add_argument('--dataset_path', type=str,default='/media/wangyz/DATA/UBUNTU_data/dataset/PT/trevi_fountain')
    parser.add_argument('--resolution', type=int, default=8)
    parser.add_argument('--sam_ckpt_path', type=str, default="ckpts/sam_vit_h_4b8939.pth")
    parser.add_argument('--dataset_type', type=str, default='pt')
    # parser.add_argument('--multi_app_seg_type',type=str,choices=['broadcast','seg-case-by-case'])
    parser.add_argument("--appearance_target_path_list", type=str,nargs='+',
                        default=['15457887_10227170235', '45182190_511249303', '80288369_2336500045'])
    parser.add_argument("--appearance_self_render",type=bool,default=True)
    parser.add_argument('--debug',action='store_true')
    parser.add_argument('--debug_gt_json_folder', default='/media/wangyz/DATA/UBUNTU_data/dataset/PT/label/brandenburg_gate')

    parser.add_argument("--iteration", default=350000, type=int)
    args = parser.parse_args()
    torch.set_default_dtype(torch.float32)

    dataset_path = args.dataset_path
    sam_ckpt_path = args.sam_ckpt_path
    debug = args.debug
    print('Debug Mode:')
    print(debug)
    # flag_save_rendering_image = False
    # if debug:
    #     flag_save_rendering_image = True
    flag_save_rendering_image = True
    if args.dataset_type == 'pt':
        img_folder = os.path.join(dataset_path, 'dense', 'images')
    elif args.dataset_type == 'on-the-go':
        img_folder = os.path.join(dataset_path, 'images')
    else:
        img_folder = os.path.join(dataset_path, 'images')
    data_list = os.listdir(img_folder)
    data_list.sort()

    model = OpenCLIPNetwork(OpenCLIPNetworkConfig)
    sam = sam_model_registry["vit_h"](checkpoint=sam_ckpt_path).to('cuda')
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.7,
        box_nms_thresh=0.7,
        stability_score_thresh=0.85,
        crop_n_layers=1,
        crop_n_points_downscale_factor=1,
        min_mask_region_area=100,
    )

    img_list = []
    WARNED = False
    for data_path in data_list:
        image_path = os.path.join(img_folder, data_path)
        image = cv2.imread(image_path)

        orig_w, orig_h = image.shape[1], image.shape[0]
        if args.resolution == -1:
            if orig_h > 1080:
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1080P), rescaling to 1080P.\n "
                          "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_h / 1080
            else:
                global_down = 1
        else:
            global_down = args.resolution

        scale = float(global_down)

        resolution = (round(orig_w / scale), round(orig_h / scale))
        image = cv2.resize(image, resolution)
        image = torch.from_numpy(image)
        img_list.append(image)
    images = [img_list[i].permute(2, 0, 1)[None, ...] for i in range(len(img_list))]
    # imgs = torch.cat(images) #dont use cat in in-the-wild setting
    if not args.debug:
        save_folder = os.path.join(dataset_path, 'language_features')
    else:
        save_folder = os.path.join(dataset_path, 'language_features_debug')
    os.makedirs(save_folder, exist_ok=True)
    print('save_folder:')
    print(save_folder)


    # render multi-appearance images
    itw_mp = itw_model.extract(args)
    itw_ap = itw_ap.extract(args)
    itw_te = itw_te.extract(args)
    pipeline = itw_pipeline.extract(args)
    print('Rendering virtual appearance...')
    itw_renderer = WEGSRenderer(itw_mp,itw_ap,itw_te,pipeline,args.iteration)
    if args.appearance_self_render:
        aug_ma_renderings = itw_renderer.batch_render_and_save_all_training_data_with_self_appearance(args.appearance_target_path_list,flag_save_rendering_image)
    else:
        aug_ma_renderings = itw_renderer.batch_render_and_save_all_training_data(args.appearance_target_path_list, flag_save_rendering_image)
    sorted_images = []
    sorted_data_list = []
    if args.dataset_type == 'pt':
        # filter the training images
        for k in aug_ma_renderings.keys():
            query_key = k + '.jpg'
            if query_key in data_list:
                index = data_list.index(query_key)
                sorted_images.append(images[index])
                sorted_data_list.append(data_list[index])
    elif args.dataset_type == 'on-the-go':
        for k in aug_ma_renderings.keys():
            query_key = k + '.JPG'
            if query_key in data_list:
                index = data_list.index(query_key)
                sorted_images.append(images[index])
                sorted_data_list.append(data_list[index])
    if debug: # filter the training_dataset in a very small scale only read the test dataset( from the debug_gt_json_folder)
        gt_json_folder = os.listdir(args.debug_gt_json_folder)
        test_img_list = [x for x in gt_json_folder if x.endswith('.jpg')]# FIXME JPG
        sorted_imgs_pre = sorted_images
        sorted_data_list_pre = sorted_data_list
        sorted_images = []
        sorted_data_list =[]
        for query_key in test_img_list:
            index = sorted_data_list_pre.index(query_key)
            sorted_images.append(sorted_imgs_pre[index])
            sorted_data_list.append(sorted_data_list_pre[index])
    print('length of dataset(sorted_images):',len(sorted_images))
    print('length of dataset(aug_ma_renderings):', len(aug_ma_renderings))
    print('length of dataset(sorted_data_list):', len(sorted_data_list))
    print('inference CLIP and SAM for each augment feature')
    # sorted_images_batch = sorted_images[859:]
    # sorted_data_list_batch = sorted_data_list[589:]
    # aug_ma_renderings_batch = aug_ma_renderings
    sorted_images_batch = sorted_images
    sorted_data_list_batch = sorted_data_list
    aug_ma_renderings_batch = aug_ma_renderings
    print('BATCH1')
    create(sorted_images_batch, sorted_data_list_batch, save_folder, aug_ma_renderings_batch, debug)
    print('Done')
