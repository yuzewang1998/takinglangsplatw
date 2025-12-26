import numpy as np
import torch
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import mediapy as media
import cv2
import colormaps
from pathlib import Path

import PIL.Image as Image
import torchvision.transforms as transforms
import torchvision
import os
def show_points(coords, labels, ax, marker_size=100):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='firebrick', marker='o',
               s=marker_size, edgecolor='black', linewidth=2.5, alpha=1)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='o',
               s=marker_size, edgecolor='black', linewidth=1.5, alpha=1)   


def show_box(boxes, ax, color=None):
    if type(color) == str and color == 'random':
        color = np.random.random(3)
    elif color is None:
        color = 'black'
    for box in boxes.reshape(-1, 4):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=color, facecolor=(0,0,0,0), lw=4, 
                                   capstyle='round', joinstyle='round', linestyle='dotted')) 


def show_result(image, point, bbox, save_path):
    plt.figure()
    plt.imshow(image)
    rect = patches.Rectangle((0, 0), image.shape[1]-1, image.shape[0]-1, linewidth=0, edgecolor='none', facecolor='white', alpha=0.3)
    plt.gca().add_patch(rect)
    input_point = point.reshape(1,-1)
    input_label = np.array([1])
    show_points(input_point, input_label, plt.gca())
    if bbox is not  None:
        show_box(bbox, plt.gca())
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.0, dpi=200)
    plt.close()


def smooth(mask):
    h, w = mask.shape[:2]
    im_smooth = mask.copy()
    scale = 3
    for i in range(h):
        for j in range(w):
            square = mask[max(0, i-scale) : min(i+scale+1, h-1),
                          max(0, j-scale) : min(j+scale+1, w-1)]
            im_smooth[i, j] = np.argmax(np.bincount(square.reshape(-1)))
    return im_smooth


def colormap_saving(image: torch.Tensor, colormap_options, save_path):
    """
    if image's shape is (h, w, 1): draw colored relevance map;
    if image's shape is (h, w, 3): return directively;
    if image's shape is (h, w, c): execute PCA and transform it into (h, w, 3).
    """
    output_image = (
        colormaps.apply_colormap(
            image=image,
            colormap_options=colormap_options,
        ).cpu().numpy()
    )
    if save_path is not None:
        media.write_image(save_path.with_suffix(".png"), output_image, fmt="png")
    return output_image


def vis_mask_save(mask, save_path: Path = None):
    mask_save = mask.copy()
    mask_save[mask == 1] = 255
    save_path.parent.mkdir(exist_ok=True, parents=True)
    cv2.imwrite(str(save_path), mask_save)


def polygon_to_mask(img_shape, points_list):
    points = np.asarray(points_list, dtype=np.int32)
    mask = np.zeros(img_shape, dtype=np.uint8)
    cv2.fillPoly(mask, [points], 1)
    return mask


def stack_mask(mask_base, mask_add):
    mask = mask_base.copy()
    mask[mask_add != 0] = 1
    return mask


# def add_text_to_highlighted_area(image_tensor, mask_tensor, text,
#                                  thickness=2, alpha=0.7, bg_alpha=0.3,
#                                  padding_ratio=0.1, corner_radius=5):
#     """
#     优化后的文本标注函数，包含以下改进：
#     1. 自动调整字体大小
#     2. 半透明圆角背景
#     3. 实心白色字体
#     4. 更智能的文本定位
#
#     :param image_tensor: 原始图像Tensor (3, H, W)
#     :param mask_tensor: 高光区域掩码Tensor (1, H, W)
#     :param text: 要添加的文本
#     :param thickness: 字体粗细
#     :param alpha: 文本不透明度 (1为完全不透明)
#     :param bg_alpha: 背景不透明度 (0-1)
#     :param padding_ratio: 背景与文本的间距比例（相对于字体大小）
#     :param corner_radius: 圆角半径（像素）
#     :return: 处理后的图像Tensor
#     """
#     # 转换为NumPy数组
#     image_np = (image_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
#     image_np = np.ascontiguousarray(image_np)
#     h, w = image_np.shape[:2]
#
#     # 处理mask
#     mask_np = mask_tensor.squeeze(0).cpu().numpy()
#     y_indices, x_indices = np.where(mask_np > 0)
#     if len(y_indices) == 0:
#         return transforms.ToTensor()(image_np)
#
#     # 计算高光区域边界
#     min_x, max_x = np.min(x_indices), np.max(x_indices)
#     min_y, max_y = np.min(y_indices), np.max(y_indices)
#     region_width = max_x - min_x
#     region_height = max_y - min_y
#
#     # 动态计算字体大小（根据区域尺寸）
#     base_scale = min(region_width, region_height) / 150  # 调整分母可改变缩放比例
#     font_scale = np.clip(base_scale, 0.5, 2.0)  # 限制最小0.5，最大2.0
#
#     # 获取文本尺寸
#     font = cv2.FONT_HERSHEY_SIMPLEX  # 最接近Arial的OpenCV字体
#     (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
#
#     # 计算文本位置（优先置于区域上方）
#     padding = int(text_height * padding_ratio)
#     text_x = min_x + int(0.05 * region_width)  # 向右偏移5%区域宽度
#     text_y = min_y - padding
#
#     # 如果上方空间不足，改为放在区域下方
#     if text_y - text_height - 2 * padding < 0:
#         text_y = max_y + text_height + 2 * padding
#         if text_y > h - padding:
#             text_y = h - padding
#
#     # 创建文本和背景的叠加层
#     overlay = image_np.copy()
#
#     # 绘制圆角背景矩形
#     bg_top_left = (text_x - padding, text_y - text_height - padding)
#     bg_bottom_right = (text_x + text_width + padding, text_y + padding)
#
#     # 确保坐标在图像范围内
#     bg_top_left = (max(0, bg_top_left[0]), max(0, bg_top_left[1]))
#     bg_bottom_right = (min(w - 1, bg_bottom_right[0]), min(h - 1, bg_bottom_right[1]))
#
#     # 圆角矩形绘制函数
#     def draw_rounded_rect(img, top_left, bottom_right, color, radius, alpha):
#         overlay = img.copy()
#         x1, y1 = top_left
#         x2, y2 = bottom_right
#
#         # 绘制圆角
#         cv2.ellipse(overlay, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, -1)
#         cv2.ellipse(overlay, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, -1)
#         cv2.ellipse(overlay, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, -1)
#         cv2.ellipse(overlay, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, -1)
#
#         # 绘制矩形主体
#         cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, -1)
#         cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, -1)
#
#         # 混合叠加层
#         cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
#
#     # 绘制半透明黑色背景
#     draw_rounded_rect(image_np, bg_top_left, bg_bottom_right, (0, 0, 0), corner_radius, bg_alpha)
#
#     # 绘制白色实心文本
#     cv2.putText(image_np, text, (text_x, text_y), font,
#                 font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
#
#     # 混合文本层
#     image_np = cv2.addWeighted(image_np, alpha, overlay, 1 - alpha, 0)
#
#     return transforms.ToTensor()(image_np)


def add_text_to_highlighted_area(image_tensor, mask_tensor, text, font_scale=1, thickness=2, alpha=1.0):
    """
    在高光边缘旁边添加文本，并根据 alpha 值调整文本透明度。

    :param image_tensor: PyTorch Tensor, 原始 RGB 图像 (3, H, W)
    :param mask_tensor: PyTorch Tensor, 高光边缘的掩码 (1, H, W)
    :param text: str, 要添加的文本
    :param font_scale: float, 字体大小
    :param thickness: int, 文字描边厚度
    :param alpha: float, 文本透明度 (0-1)
    :return: 处理后的 PyTorch Tensor
    """
    # 转换为 NumPy 数组，确保值范围是 [0, 255] 且数据类型是 np.uint8
    image_np = (image_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    image_np = np.ascontiguousarray(image_np)  # 确保数组是连续的

    # 确保 mask 是 2D
    mask_np = mask_tensor.squeeze(0).cpu().numpy()  # 从 (1, H, W) 到 (H, W)

    # 获取高光区域坐标
    y_indices, x_indices = np.where(mask_np > 0)
    if len(y_indices) == 0 or len(x_indices) == 0:
        return transforms.ToTensor()(image_np)  # 没有高光区域，直接返回原图

    # 计算边界框
    min_x, max_x = np.min(x_indices), np.max(x_indices)
    min_y, max_y = np.min(y_indices), np.max(y_indices)

    # 文字位置（默认放在物体左上角）
    text_x, text_y = int(min_x), int(min_y) - 10  # 转换为整数坐标
    if text_y < 20:  # 避免文字超出顶部
        text_y = int(min_y) + 20

    # 创建临时图层绘制文本
    text_overlay = np.zeros_like(image_np)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # 绘制白色描边和黑色字体到临时图层

    cv2.putText(text_overlay, text, (text_x, text_y), font,
                font_scale, (25, 202, 173), thickness, cv2.LINE_AA)
    # 生成文本区域的掩码（非黑色像素）
    mask = (text_overlay != 0).any(axis=2)

    # 混合临时图层和原图（仅影响文本区域）
    image_np[mask] = ((1-alpha) * text_overlay[mask] + ( alpha) * image_np[mask]).astype(np.uint8)

    # 返回为 Tensor
    return transforms.ToTensor()(image_np)

def mask2highlight(mask_tensor,highlight_color=(255, 255, 0), thickness=4):
    """
    对 mask_tensor 提取边缘，并在边缘区域添加高光，同时避免黑边。

    :param mask_tensor: PyTorch Tensor, 二值掩码 (1, H, W)
    :param highlight_color: 高光颜色 (B, G, R)，默认为黄色 (255, 255, 0)
    :param thickness: 高光边缘的厚度
    :return: 处理后的边缘高光图像 (PyTorch Tensor)
    """
    mask_np = (mask_tensor.squeeze().cpu().numpy() * 255).astype(np.uint8)

    # 进行边缘检测
    edges = cv2.Canny(mask_np, 100, 200)

    # 扩展边缘厚度
    kernel = np.ones((thickness, thickness), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)

    # 创建一个白色背景的高光层
    highlight_layer = np.ones((mask_np.shape[0], mask_np.shape[1], 3), dtype=np.uint8) * 255  # 先填充为白色

    # 只在边缘区域添加颜色
    for i in range(3):
        highlight_layer[:, :, i] = np.where(edges_dilated > 0, highlight_color[i], 255)  # 只修改边缘像素

    # 进行高斯模糊（此时黑色已经变成了白色，所以不会渗透）
    highlight_layer = cv2.GaussianBlur(highlight_layer, (3, 3), sigmaX=2)

    # **关键改进**: 恢复背景透明，即去掉非边缘的白色
    for i in range(3):
        highlight_layer[:, :, i] = np.where(edges_dilated > 0, highlight_layer[:, :, i], 0)

    # 转换为 PyTorch Tensor
    edge_highlight_tensor = transforms.ToTensor()(Image.fromarray(highlight_layer))

    return edge_highlight_tensor
if __name__ == "__main__":


    image_path = '/home/wangyz/Downloads/rgb.png'
    mask_path = '/home/wangyz/Downloads/mask.png'
    transform_image = transforms.Compose([
        transforms.ToTensor(),  # 转为 [C, H, W] 张量，值范围 [0, 1]
    ])
    transform_mask = transforms.Compose([
        transforms.ToTensor(),  # 转为 [1, H, W] 张量，值范围 [0, 1]
    ])
    image = Image.open(image_path).convert('RGB')
    mask = Image.open(mask_path).convert('L')
    image_tensor = transform_image(image)  # [3, H, W]
    mask_tensor = transform_mask(mask)  # [1, H, W]

    if mask_tensor.shape[1:] != image_tensor.shape[1:]:
        mask_tensor = transforms.Resize(image_tensor.shape[1:])(mask_tensor)

    mask_highlight_tensor = mask2highlight(mask_tensor)

    highlight_mask = mask_highlight_tensor.sum(dim=0, keepdim=True) > 0  # 只选取有高光的地方
    fused_tensor = image_tensor.clone()

    # 只影响高光区域
    fused_tensor[:, highlight_mask[0, :, :]] = (
            alpha * fused_tensor[:, highlight_mask[0, :, :]] +
            (1 - alpha) * mask_highlight_tensor[:, highlight_mask[0, :, :]]
    ).clamp(0, 1)  # 确保值在 0-1 之间

    # alpha = 0.5
    # fused_tensor = alpha * image_tensor + (1 - alpha) * mask_highlight_tensor
    # fused_tensor = fused_tensor.clamp(0, 1)

    torchvision.utils.save_image(fused_tensor, os.path.join('/home/wangyz/Downloads', '{0:05d}'.format(0000) + ".png"))
