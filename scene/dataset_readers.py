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
import math
import os
import sys
from PIL import Image
import open3d as o3d
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    depth_params: dict
    image_path: str
    image_name: str
    depth_path: str
    width: int
    height: int
    is_test: bool
    K: np.array=None


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    is_nerf_synthetic: bool

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, depths_params, images_folder,
                      depths_folder, test_cam_names_list, detector):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model == "SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "PINHOLE" or intr.model=="OPENCV":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"
        n_remove = len(extr.name.split('.')[-1]) + 1
        depth_params = None
        if depths_params is not None:
            try:
                depth_params = depths_params[extr.name[:-n_remove]]
            except:
                print("\n", key, "not found in depths_params")

        image_path = os.path.join(images_folder, extr.name)
        if detector == 'DexiNed':
            edge_path = image_path.replace('images', 'edge_DexiNed')
            edge_path = edge_path.replace('.jpg', '.png')
            image = Image.open(edge_path)
            # from PIL import ImageOps
            # image = ImageOps.invert(image)
        else:
            edge_path = image_path.replace('images', 'edge_PidiNet')
            edge_path = edge_path.replace('.jpg', '.png')
            image = Image.open(edge_path)

        image_name = extr.name.replace('.jpg', '.png')
        depth_path = os.path.join(depths_folder, f"{extr.name[:-n_remove]}.png") if depths_folder != "" else ""
        cx = width / 2.0
        cy = height / 2.0
        K = np.array([
            [focal_length_x, 0, cx],
            [0, focal_length_y, cy],
            [0, 0, 1]
        ])

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image, depth_params=depth_params,
                              image_path=image_path, image_name=image_name, depth_path=depth_path,
                              width=width, height=height, is_test=image_name in test_cam_names_list, K=K)
        cam_infos.append(cam_info)

    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, depths, eval, train_test_exp, llffhold=8, detector='DexiNed'):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    depth_params_file = os.path.join(path, "sparse/0", "depth_params.json")
    ## if depth_params_file isnt there AND depths file is here -> throw error
    depths_params = None
    if depths != "":
        try:
            with open(depth_params_file, "r") as f:
                depths_params = json.load(f)
            all_scales = np.array([depths_params[key]["scale"] for key in depths_params])
            if (all_scales > 0).sum():
                med_scale = np.median(all_scales[all_scales > 0])
            else:
                med_scale = 0
            for key in depths_params:
                depths_params[key]["med_scale"] = med_scale

        except FileNotFoundError:
            print(f"Error: depth_params.json file not found at path '{depth_params_file}'.")
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred when trying to open depth_params.json file: {e}")
            sys.exit(1)

    if eval:
        if "360" in path:
            llffhold = 8
        if llffhold:
            print("------------LLFF HOLD-------------")
            cam_names = [cam_extrinsics[cam_id].name for cam_id in cam_extrinsics]
            cam_names = sorted(cam_names)
            test_cam_names_list = [name for idx, name in enumerate(cam_names) if idx % llffhold == 0]
        else:
            with open(os.path.join(path, "sparse/0", "test.txt"), 'r') as file:
                test_cam_names_list = [line.strip() for line in file]
    else:
        test_cam_names_list = []

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(
        cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, depths_params=depths_params,
        images_folder=os.path.join(path, reading_dir), 
        depths_folder=os.path.join(path, depths) if depths != "" else "",
        test_cam_names_list=test_cam_names_list, detector=detector)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    train_cam_infos = [c for c in cam_infos] #[c for c in cam_infos if train_test_exp or not c.is_test]
    test_cam_infos = [c for c in cam_infos if c.is_test]

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
   
    print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
    try:
        points, rgb, _ = read_points3D_binary(bin_path)
    except:
        points, rgb, _ = read_points3D_text(txt_path)

  
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           is_nerf_synthetic=False)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, depths_folder='', white_background=False, 
                              is_test=False, extension=".png", detector='DexiNed'):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)
            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1
            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            edge_path = image_path.replace('ABC-NEF/', 'ABC-NEF_Edge/data/')
            edge_path = edge_path.replace('train', 'edge_'+ detector)
            image = Image.open(edge_path)
            image = image.convert("RGBA")
            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy
            FovX = fovx

            depth_path = os.path.join(depths_folder, f"{image_name}.png") if depths_folder != "" else ""
            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name,
                            width=image.size[0], height=image.size[1], depth_path=depth_path, depth_params=None, is_test=is_test))
            
    return cam_infos


def readEMAP(path, transformsfile, depths_folder, white_background, is_test, extension=".png",
                              detector='DexiNed'):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as f:
        meta_data = json.load(f)
        height = meta_data["height"]
        width = meta_data["width"]
        frames = meta_data["frames"]
        bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["rgb_path"])
            c2w = np.array(frame["camtoworld"])
            Kmat = np.array(frame["intrinsics"])
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, 'color',  frame["rgb_path"])
            image_name = Path(cam_name).stem
            if detector=='PidiNet':
                edge_path = image_path.replace('/color', '/edge_PidiNet')
                image = Image.open(edge_path)
            elif detector=='DexiNed':
                edge_path = image_path.replace('/color', '/edge_DexiNed')
                image = Image.open(edge_path)
            else:
                assert False, f"Detector {detector} not supported"

            image = image.convert("RGBA")
            FovY = focal2fov(Kmat[1, 1], image.size[1])
            FovX = focal2fov(Kmat[0, 0], image.size[0])

            depth_path = os.path.join(depths_folder, f"{image_name}.png") if depths_folder != "" else ""
            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                        image_path=image_path, image_name=image_name,
                                        width=image.size[0], height=image.size[1], depth_path=depth_path,
                                        depth_params=None, is_test=is_test, K=Kmat))

    return cam_infos

def readNerfSyntheticInfo(path, white_background, depths, eval, extension=".png",
                          detector='DexiNed'):

    depths_folder=os.path.join(path, depths) if depths != "" else ""
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", depths_folder,
                                                white_background, False, extension, detector)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", depths_folder,
                                               white_background, True, extension, detector)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if 'ABC' in path:
        # We create random points inside the bounds of the synthetic Blender scenes
        # xyz = np.random.random((num_pts, 3)) * 1.1 - 0.05
        num_pts_per_axis = 15
        num_pts =  num_pts_per_axis**3
        print(f"Generating random point cloud ({num_pts})...")
        x = np.linspace(-0.05, 1.05, num_pts_per_axis)
        y = np.linspace(-0.05, 1.05, num_pts_per_axis)
        z = np.linspace(-0.05, 1.05, num_pts_per_axis)
        xx, yy, zz = np.meshgrid(x, y, z)
        xyz = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T
        shs = np.random.random((num_pts, 3)) / 255.0
    else:
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
    pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

    storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           is_nerf_synthetic=True)
    return scene_info


def rendemapInfo(path, white_background=False, depths='', eval=False,
                 extension=".png", detector='DexiNed', init_random_init=True):
    depths_folder = os.path.join(path, depths) if depths != "" else ""
    print("Reading Training Transforms")
    train_cam_infos = readEMAP(path, "meta_data.json", depths_folder,
                                                white_background, False, extension, detector)
    print("Reading Test Transforms")
    test_cam_infos = readEMAP(path, "meta_data.json", depths_folder,
                                               white_background, True, extension, detector)

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    num_pts = 8000+1

    if init_random_init:
        num_pts_per_axis = 15
        num_pts =  num_pts_per_axis**3
        print(f"Generating random point cloud ({num_pts})...")
        x = np.linspace(-0.05, 1.05, num_pts_per_axis)
        y = np.linspace(-0.05, 1.05, num_pts_per_axis)
        z = np.linspace(-0.05, 1.05, num_pts_per_axis)
        xx, yy, zz = np.meshgrid(x, y, z)
        xyz = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T
        
    else:
        seed_points_path = os.path.join(path, 'sparse_sfm_points.txt')
        if seed_points_path.endswith(".txt"):
            try:
                xyz = (np.loadtxt(seed_points_path))
            except:
                points3d = read_points3D_text(seed_points_path)
                xyz = np.array([points3d[point].xyz for point in points3d])

        elif seed_points_path.endswith(".ply"):
            sparse_pc = o3d.io.read_point_cloud(seed_points_path)
            xyz = np.asarray(sparse_pc.points).reshape(-1, 3)
        num_seed_points = xyz.shape[0]
        if num_seed_points < num_pts:
            num_sample_more = num_pts - num_seed_points
            replication_factor = int(np.ceil(num_sample_more / num_seed_points))
            noise = 0.1 * np.random.random((replication_factor * num_seed_points, 3))
            seed_points_extra = np.concatenate([xyz] * replication_factor, axis=0) + noise
            xyz = np.concatenate([xyz, seed_points_extra], axis=0)

        else:
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(xyz)
            voxel_size = 0.05  
            downsampled_cloud = point_cloud.voxel_down_sample(voxel_size)
            xyz = np.asarray(downsampled_cloud.points)
    num_pts = xyz.shape[0]
    shs = np.random.random((num_pts, 3)) / 255.0
    pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
    
    storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           is_nerf_synthetic=True)
    return scene_info


sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "emap": rendemapInfo,
}