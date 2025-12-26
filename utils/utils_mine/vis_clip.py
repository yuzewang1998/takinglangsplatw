import numpy as np
import torch
import matplotlib.pyplot as plt

# 加载数据（路径需根据实际情况修改）
f1_path = '/home/wangyz/Downloads/test/language_features/1_f.npy'
s1_path = '/home/wangyz/Downloads/test/language_features/1_s.npy'
f1 = np.load(f1_path)
s1 = np.load(s1_path)
f2_path = '/home/wangyz/Downloads/test/language_features/2_f.npy'
s2_path = '/home/wangyz/Downloads/test/language_features/2_s.npy'
f2 = np.load(f2_path)
s2 = np.load(s2_path)
f3_path = '/home/wangyz/Downloads/test/language_features/3_f.npy'
s3_path = '/home/wangyz/Downloads/test/language_features/3_s.npy'
f3 = np.load(f3_path)
s3 = np.load(s3_path)
f4_path = '/home/wangyz/Downloads/test/language_features/4_f.npy'
s4_path = '/home/wangyz/Downloads/test/language_features/4_s.npy'
f4 = np.load(f4_path)
s4 = np.load(s4_path)
# 转换为Tensor并提取样本
feature_map1 = torch.Tensor(f1[s1])[1]  # 形状假设为 [H, W, C]
feature_map2 = torch.Tensor(f2[s2])[1]
feature_map3 = torch.Tensor(f3[s3])[1]
feature_map4 = torch.Tensor(f4[s4])[1]
# 对每个通道独立归一化
def normalize_channels(feature_map):
    normalized = torch.zeros_like(feature_map)
    for c in range(feature_map.shape[2]):
        channel = feature_map[:, :, c]
        min_val = channel.min()
        max_val = channel.max()
        normalized[:, :, c] = (channel - min_val) / (max_val - min_val + 1e-8)
    return normalized

feature_map1_normalized = normalize_channels(feature_map1)
feature_map2_normalized = normalize_channels(feature_map2)
feature_map3_normalized = normalize_channels(feature_map3)
feature_map4_normalized = normalize_channels(feature_map4)
# 提取前三个通道并转换为numpy数组
sample1 = feature_map1_normalized[:, :, :3].cpu().numpy()
sample2 = feature_map2_normalized[:, :, :3].cpu().numpy()
sample3 = feature_map3_normalized[:, :, :3].cpu().numpy()
sample4 = feature_map4_normalized[:, :, :3].cpu().numpy()
# 可视化（不再使用cmap）
# fig, axes = plt.subplots(1, 2, figsize=(12, 6))
# axes[0].imshow(sample1)
# axes[0].set_title('Feature Map 1')
# axes[0].axis('off')
#
# axes[1].imshow(sample2)
# axes[1].set_title('Feature Map 2')
# axes[1].axis('off')

plt.show()
# 保存路径设置（根据你的需求修改路径）
save_path1 = '/home/wangyz/Downloads/test/feature_map1.png'  # 第一张图保存路径
save_path2 = '/home/wangyz/Downloads/test/feature_map2.png'  # 第二张图保存路径
save_path3 = '/home/wangyz/Downloads/test/feature_map3.png'  # 第一张图保存路径
save_path4 = '/home/wangyz/Downloads/test/feature_map4.png'  # 第二张图保存路径
# 保存 Feature Map 1（独立高清图）
plt.figure(figsize=(6, 6), dpi=300)  # 控制分辨率和尺寸
plt.imshow(sample1)
plt.axis('off')
plt.savefig(save_path1, bbox_inches='tight', pad_inches=0, transparent=True)  # 透明背景
plt.close()  # 关闭当前figure释放内存

# 保存 Feature Map 2（独立高清图）
plt.figure(figsize=(6, 6), dpi=300)  # 控制分辨率和尺寸
plt.imshow(sample2)
plt.axis('off')
plt.savefig(save_path2, bbox_inches='tight', pad_inches=0, transparent=True)  # 透明背景
plt.close()

# 保存 Feature Map 2（独立高清图）
plt.figure(figsize=(6, 6), dpi=300)  # 控制分辨率和尺寸
plt.imshow(sample3)
plt.axis('off')
plt.savefig(save_path3, bbox_inches='tight', pad_inches=0, transparent=True)  # 透明背景
plt.close()
# 保存 Feature Map 2（独立高清图）
plt.figure(figsize=(6, 6), dpi=300)  # 控制分辨率和尺寸
plt.imshow(sample4)
plt.axis('off')
plt.savefig(save_path4, bbox_inches='tight', pad_inches=0, transparent=True)  # 透明背景
plt.close()
print(f"特征图已保存至：\n{save_path1}\n{save_path2}")