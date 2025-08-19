
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import json

from skimage.measure import LineModelND, ransac
from scipy.spatial.distance import cdist
from scipy.optimize import curve_fit
from scipy.optimize import minimize

##### Taken from EMAP #####

def bezier_curve(tt, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11):
    n = len(tt)
    matrix_t = np.concatenate(
        [(tt**3)[..., None], (tt**2)[..., None], tt[..., None], np.ones((n, 1))],
        axis=1,
    ).astype(float)
    matrix_w = np.array(
        [[-1, 3, -3, 1], [3, -6, 3, 0], [-3, 3, 0, 0], [1, 0, 0, 0]]
    ).astype(float)
    matrix_p = np.array(
        [[p0, p1, p2], [p3, p4, p5], [p6, p7, p8], [p9, p10, p11]]
    ).astype(float)
    return np.dot(np.dot(matrix_t, matrix_w), matrix_p).reshape(-1)

def line_fitting(endpoints):
    center = np.mean(endpoints, axis=0)

    # compute the main direction through SVD
    endpoints_centered = endpoints - center
    u, s, vh = np.linalg.svd(endpoints_centered, full_matrices=False)
    lamda = s[0] / np.sum(s)
    main_direction = vh[0]
    main_direction = main_direction / np.linalg.norm(main_direction)

    # project endpoints onto the main direction
    projections = []
    for endpoint_centered in endpoints_centered:
        projections.append(np.dot(endpoint_centered, main_direction))
    projections = np.array(projections)

    # construct final line
    straight_line = np.zeros(6)
    # print(np.min(projections), np.max(projections))
    straight_line[:3] = center + main_direction * np.min(projections)
    straight_line[3:] = center + main_direction * np.max(projections)

    return straight_line, lamda

def bezier_fit(xyz, error_threshold=0.02):
    n = len(xyz)
    t = np.linspace(0, 1, n)
    xyz = xyz.reshape(-1)

    popt, _ = curve_fit(bezier_curve, t, xyz)

    # Generate fitted curve
    fitted_curve = bezier_curve(t, *popt).reshape(-1, 3)

    # Calculate residuals
    residuals = xyz.reshape(-1, 3) - fitted_curve

    # Calculate RMSE
    rmse = np.sqrt(np.mean(np.sum(residuals**2, axis=1)))

    if rmse > error_threshold:
        return None
    else:
        return popt


def fit_straight_line(points):
    """拟合直线，返回直线的起点和终点"""
    # 使用最小二乘法拟合直线
    mean_point = np.mean(points, axis=0)
    centered_points = points - mean_point
    cov_matrix = np.dot(centered_points.T, centered_points) / len(points)
    # 计算协方差矩阵的特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # 方向向量是最大特征值对应的特征向量
    direction = eigenvectors[:, np.argmax(eigenvalues)]
    direction /= np.linalg.norm(direction)  # 单位化

    # 将所有点投影到直线上
    projections = np.dot(points - mean_point, direction)

    # 找到投影的最小和最大值
    t_min = np.min(projections)
    t_max = np.max(projections)

    # 计算线段的起点和终点
    start = mean_point + t_min * direction
    end = mean_point + t_max * direction
    return start, end, direction, mean_point, t_min, t_max