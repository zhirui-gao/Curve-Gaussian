import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import seaborn as sns
from matplotlib import colors as mplcolors


def get_fancy_cmap():
    colors = sns.color_palette('hls', 100)
    gold = mplcolors.to_rgb('gold')
    colors = [gold] + colors[3:] + colors[:2]
    raw_cmap = mplcolors.LinearSegmentedColormap.from_list('Custom', colors)

    def cmap(values):
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
        return raw_cmap(values)[:, :3]

    return cmap


def get_fancy_color(num):
    values = torch.linspace(0, 1, num + 1)[1:]
    colors = torch.from_numpy(get_fancy_cmap()(values.cpu().numpy())).float()
    return colors


def visualize_dirs_pca(vis_dir, title="Direction PCA Visualization"):
    """
    在 2D 图像中可视化 3D 方向向量 (prior_normal 或 gradient_direction)
    使用 PCA 降维到单通道，并用颜色表示方向变化

    Args:
        vis_dir (torch.Tensor): 形状为 (3, H, W) 的方向张量 (CUDA tensor)
        title (str): 图像标题
    """
    # 转换为 NumPy 并展平
    vis_dir_np = vis_dir.detach().cpu().numpy()  # (3, H, W)
    C, H, W = vis_dir_np.shape
    vis_dir_flat = vis_dir_np.reshape(C, -1).T  # 变成 (H*W, 3)

    # 进行 PCA 降维 (3D -> 1D)
    pca = PCA(n_components=1)
    vis_pca = pca.fit_transform(vis_dir_flat)  # (H*W, 1)
    vis_pca = vis_pca.reshape(H, W)  # 重新 reshape 成 (H, W)

    # 归一化到 0-1
    vis_pca = (vis_pca - vis_pca.min()) / (vis_pca.max() - vis_pca.min())

    # 可视化
    plt.figure(figsize=(8, 6))
    plt.imshow(vis_pca, cmap="jet", interpolation="nearest")
    plt.colorbar(label="PCA Reduced Value")
    plt.title(title)
    plt.xlabel("Width")
    plt.ylabel("Height")
    plt.show()

# 示例数据








def visualize_magnitude(magnitude, title="Gradient Magnitude"):
    """
    可视化梯度强度 (gradient_magnitude) 的 2D 热力图

    Args:
        gradient_magnitude (torch.Tensor): 形状为 (H, W) 的梯度强度张量 (CUDA tensor)
        title (str): 图像标题
    """
    # 将 CUDA Tensor 转换为 NumPy
    grad_mag_np = magnitude.detach().cpu().numpy()

    # 可视化
    plt.figure(figsize=(8, 6))
    plt.imshow(grad_mag_np, cmap="jet", interpolation="nearest")
    plt.colorbar(label="Gradient Magnitude")
    plt.title(title)
    plt.xlabel("Width")
    plt.ylabel("Height")
    plt.show()

