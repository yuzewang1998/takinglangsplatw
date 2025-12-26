import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import argparse

def load_and_process_mask(mask_path):
    """
    加载mask图像并转换为二值化mask
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"警告: 无法加载mask文件 {mask_path}")
        return None
    
    # 二值化处理，白色区域为前景
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return binary_mask

def overlay_masks_on_image(rgb_image_path, mask_paths, output_path=None, alpha=0.5):
    """
    将多个mask按照不同颜色叠加到RGB图像上
    
    Args:
        rgb_image_path: 原RGB图像路径
        mask_paths: mask图像路径列表
        output_path: 输出图像路径（可选）
        alpha: 透明度 (0-1)
    """
    # 读取原RGB图像
    rgb_image = cv2.imread(rgb_image_path)
    if rgb_image is None:
        print(f"错误: 无法加载RGB图像 {rgb_image_path}")
        return None
    
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
    result_image = rgb_image.copy().astype(np.float32)
    
    # 预定义颜色列表 (RGB格式)
    colors = [
        (255, 0, 0),      # 红色
        (0, 255, 0),      # 绿色  
        (0, 0, 255),      # 蓝色
        (255, 255, 0),    # 黄色
        (255, 0, 255),    # 洋红
        (0, 255, 255),    # 青色
        (255, 165, 0),    # 橙色
        (128, 0, 128),    # 紫色
        (255, 192, 203),  # 粉色
        (0, 128, 0),      # 深绿色
    ]
    
    # 创建总的mask区域（所有mask的合并）
    total_mask = np.zeros(rgb_image.shape[:2], dtype=np.uint8)
    
    print(f"正在处理 {len(mask_paths)} 个mask...")
    
    for i, mask_path in enumerate(mask_paths):
        mask = load_and_process_mask(mask_path)
        if mask is None:
            continue
            
        # 确保mask尺寸与RGB图像相同
        if mask.shape[:2] != rgb_image.shape[:2]:
            mask = cv2.resize(mask, (rgb_image.shape[1], rgb_image.shape[0]))
        
        # 更新总mask
        total_mask = np.logical_or(total_mask, mask > 127)
        
        # 获取颜色（循环使用颜色列表）
        color = colors[i % len(colors)]
        
        # 创建彩色mask，只在白色区域叠加颜色
        mask_indices = mask > 127  # 白色区域为前景
        
        # 为每个颜色通道创建叠加
        for c in range(3):
            # 只在mask的白色区域叠加颜色
            result_image[:, :, c][mask_indices] = (
                result_image[:, :, c][mask_indices] * (1 - alpha) + 
                color[c] * alpha
            )
        
        # 获取mask文件名（用于显示）
        mask_name = Path(mask_path).stem
        print(f"  - 处理了 {mask_name}，使用颜色 RGB{color}")
    
    # 给背景（非mask区域）添加暗化蒙版
    background_dim_factor = 0.4  # 背景暗化程度，0.4表示变暗到40%
    background_indices = ~total_mask  # 非mask区域
    
    for c in range(3):
        result_image[:, :, c][background_indices] = (
            result_image[:, :, c][background_indices] * background_dim_factor
        )
    
    print(f"  - 已对背景应用 {int((1-background_dim_factor)*100)}% 暗化蒙版")
    
    # 转换回uint8格式
    result_image = np.clip(result_image, 0, 255).astype(np.uint8)
    
    # 显示结果
    plt.figure(figsize=(15, 5))
    
    # 原图
    plt.subplot(1, 3, 1)
    plt.imshow(rgb_image)
    plt.title('原RGB图像')
    plt.axis('off')
    
    # 叠加结果
    plt.subplot(1, 3, 2)
    plt.imshow(result_image)
    plt.title('Mask叠加结果')
    plt.axis('off')
    
    # 显示颜色对应关系
    plt.subplot(1, 3, 3)
    legend_image = np.ones((len(mask_paths) * 50, 200, 3), dtype=np.uint8) * 255
    
    for i, mask_path in enumerate(mask_paths):
        color = colors[i % len(colors)]
        y_start = i * 50
        y_end = (i + 1) * 50
        legend_image[y_start:y_end, :50] = color
        
        mask_name = Path(mask_path).stem.replace('chosen_', '')
        plt.text(60, y_start + 25, mask_name, fontsize=10, va='center')
    
    plt.imshow(legend_image)
    plt.title('颜色对应关系')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # 保存结果（如果指定了输出路径）
    if output_path:
        result_bgr = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, result_bgr)
        print(f"结果已保存到: {output_path}")
    
    return result_image

def main():
    """
    主函数 - 处理您的具体文件
    """
    # 您的文件路径
    workspace_dir = "/home/wangyz/Downloads/workspace"
    rgb_image_path = os.path.join(workspace_dir, "IMG_6272.JPG")
    
    mask_files = [
        "chosen_bicycles.png",
        "chosen_cobblestones.png", 
        "chosen_tree.png"
    ]
    
    mask_paths = [os.path.join(workspace_dir, mask_file) for mask_file in mask_files]
    
    # 检查文件是否存在
    if not os.path.exists(rgb_image_path):
        print(f"错误: 找不到RGB图像文件 {rgb_image_path}")
        return
    
    existing_masks = []
    for mask_path in mask_paths:
        if os.path.exists(mask_path):
            existing_masks.append(mask_path)
        else:
            print(f"警告: 找不到mask文件 {mask_path}")
    
    if not existing_masks:
        print("错误: 没有找到任何有效的mask文件")
        return
    
    print(f"找到 {len(existing_masks)} 个有效的mask文件")
    
    # 执行叠加
    output_path = os.path.join(workspace_dir, "overlay_result.png")
    overlay_masks_on_image(rgb_image_path, existing_masks, output_path, alpha=0.6)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='将多个mask按照不同颜色叠加到RGB图像上')
    parser.add_argument('--rgb_image', type=str, help='RGB图像路径')
    parser.add_argument('--mask_dir', type=str, help='mask文件目录')
    parser.add_argument('--output', type=str, help='输出文件路径')
    parser.add_argument('--alpha', type=float, default=0.6, help='透明度 (0-1)')
    
    args = parser.parse_args()
    
    if args.rgb_image and args.mask_dir:
        # 自定义模式
        mask_files = [f for f in os.listdir(args.mask_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        mask_paths = [os.path.join(args.mask_dir, f) for f in mask_files]
        
        output_path = args.output if args.output else "overlay_result.png"
        overlay_masks_on_image(args.rgb_image, mask_paths, output_path, args.alpha)
    else:
        # 使用默认设置处理您的文件
        main()
