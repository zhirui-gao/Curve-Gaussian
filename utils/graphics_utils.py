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

import torch
import math
import numpy as np
from typing import NamedTuple

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P



def getProjectionMatrixFromIntrinsics(znear, zfar, K_new, W, H):
    fx = K_new[0, 0]
    fy = K_new[1, 1]
    cx = K_new[0, 2]
    cy = K_new[1, 2]

    # Compute near plane bounds
    l = -cx * (znear / fx)
    r = (W - cx) * (znear / fx)
    b = -cy * (znear / fy)
    t = (H - cy) * (znear / fy)

    # Construct projection matrix (OpenGL/Vulkan style)
    P = torch.zeros((4, 4), dtype=torch.float32)

    P[0, 0] = 2 * znear / (r - l)
    P[0, 2] = (r + l) / (r - l)
    P[1, 1] = 2 * znear / (t - b)
    P[1, 2] = (t + b) / (t - b)
    P[2, 2] = zfar / (zfar - znear)
    P[2, 3] = -zfar * znear / (zfar - znear)
    P[3, 2] = 1.0  # Vulkan convention (use -1.0 for OpenGL)

    return P

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def quaternion_multiply(q1, q2):
    """
    四元数乘法，支持梯度回传。
    参数:
    q1 -- 第一个四元数，形状为 [B, 4] 的张量，其中 B 是批量大小
    q2 -- 第二个四元数，形状为 [B, 4] 的张量

    返回:
    结果四元数，形状为 [B, 4] 的张量
    """
    # 确保输入张量是浮点类型
    # 展开四元数
    w1, x1, y1, z1 = q1[..., 0:1], q1[..., 1:2], q1[..., 2:3], q1[..., 3:4]
    w2, x2, y2, z2 = q2[..., 0:1], q2[..., 1:2], q2[..., 2:3], q2[..., 3:4]

    # 四元数乘法的计算
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    # 合并结果
    q = torch.cat((w, x, y, z), dim=-1)
    return q

def rotate_point_by_quaternion(q, p):
    """
    通过四元数对三维点进行旋转。
    参数:
    q -- 旋转四元数，形状为 [B, 4] 的张量
    p -- 三维点，形状为 [B, 3] 的张量

    返回:
    旋转后的点，形状为 [B, 3] 的张量
    """
    # import pdb;pdb.set_trace()
    # 将点 p 扩展为四元数 p_q
    p_q = torch.cat((torch.zeros_like((p[..., :1]),device = p.device), p), dim=-1)

    # 计算四元数的逆
    q_conjugate = torch.cat((q[..., 0:1], -q[..., 1:4]), dim=-1)

    # 计算旋转后的四元数
    temp = quaternion_multiply(q, p_q)  # q * p_q
    rotated_p_q = quaternion_multiply(temp, q_conjugate)  # (q * p_q) * q^-1

    # 提取旋转后的点坐标
    rotated_p = rotated_p_q[..., 1:]

    return rotated_p