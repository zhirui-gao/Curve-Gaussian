import os
import sys
# Add the parent directory to Python path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import numpy as np
import open3d as o3d
import sklearn.neighbors as skln
from tqdm import tqdm
from scipy.io import loadmat
import multiprocessing as mp
import argparse
from pathlib import Path
import cv2
from edge_extraction.eval_utils import (
    set_random_seeds,
    get_pred_points_and_directions,
)

from scene.colmap_loader import qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary

import matplotlib.pyplot as plt

def create_video_from_images(projected_dir, original_dir, output_path, fps=10):
    """
    Create a video from projected and original images side by side using ffmpeg.
    
    Args:
        projected_dir: Directory containing projected images
        original_dir: Directory containing original images
        output_path: Path to save the output video
        fps: Frames per second
    """
    import subprocess
    import tempfile
    
    # Get all image files
    projected_files = sorted([f for f in os.listdir(projected_dir) if f.endswith('.jpg')])
    original_files = sorted([f for f in os.listdir(original_dir) if f.endswith('.jpg')])
    
    if not projected_files or not original_files:
        print("No images found in one or both directories")
        return
    
    # Create temporary directory for combined images
    with tempfile.TemporaryDirectory() as temp_dir:
        # Process each pair of images
        for i, (proj_file, orig_file) in enumerate(zip(projected_files, original_files)):
            # Read images
            proj_img = cv2.imread(os.path.join(projected_dir, proj_file))
            orig_img = cv2.imread(os.path.join(original_dir, orig_file))
            
            if proj_img is None or orig_img is None:
                print(f"Error: Could not read images {proj_file} or {orig_file}")
                continue
            
            # Resize images to same height if needed
            h1, w1 = proj_img.shape[:2]
            h2, w2 = orig_img.shape[:2]
            if h1 != h2:
                scale = h1 / h2
                orig_img = cv2.resize(orig_img, (int(w2 * scale), h1))
            
            # Combine images horizontally
            combined = np.hstack((proj_img, orig_img))
            
            # Save combined image
            temp_img_path = os.path.join(temp_dir, f'frame_{i:04d}.jpg')
            cv2.imwrite(temp_img_path, combined)
        
        # Use ffmpeg to create video
        ffmpeg_cmd = [
            'ffmpeg',
            '-y',  # Overwrite output file if it exists
            '-framerate', str(fps),
            '-i', os.path.join(temp_dir, 'frame_%04d.jpg'),
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-crf', '23',  # Constant Rate Factor (lower = better quality, 23 is default)
            output_path
        ]
        
        try:
            subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True)
            print(f"Video saved to {output_path}")
            
            # Verify the video was created
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                print("Video creation successful")
            else:
                print("Error: Video file is empty or was not created")
        except subprocess.CalledProcessError as e:
            print(f"Error creating video: {e.stderr}")
        except Exception as e:
            print(f"Unexpected error: {str(e)}")

def process_scan(
    scan_name,
    base_dir,
    exp_name,
    dataset_dir,
):
    print(f"Processing: {scan_name}")
    json_path = os.path.join(
        base_dir, scan_name, "parametric_edges.json"
    )
    if not os.path.exists(json_path):
        print(f"Invalid prediction at {scan_name}")
        return

    all_curve_points, all_line_points, all_curve_directions, \
        all_line_directions, all_curve_colors, all_line_colors, \
        num_curves, num_lines = get_pred_points_and_directions(json_path, sample_resolution=0.0005)

    all_points = (
        np.concatenate([all_curve_points, all_line_points], axis=0)
        .reshape(-1, 3)
        .astype(np.float32)
    )

    min_vals = np.min(all_points, axis=0)  # [min_x, min_y, min_z]
    max_vals = np.max(all_points, axis=0)  # [max_x, max_y, max_z]
    scale = max_vals - min_vals
    scale[scale == 0] = 1.0
    pred_colors = (
        np.concatenate([all_curve_colors, all_line_colors], axis=0)
        .reshape(-1, 3)
        .astype(np.float32)
    )

    scene_data = os.path.join(dataset_dir, scan_name)
    cameras_extrinsic_file = os.path.join(scene_data, "sparse/0", "images.bin")
    cameras_intrinsic_file = os.path.join(scene_data, "sparse/0", "cameras.bin")
    cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
    cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)

    # Project points onto each camera view and visualize
    for image_id, image in cam_extrinsics.items():
        # Get camera parameters
        camera_id = image.camera_id
        camera = cam_intrinsics[camera_id]
        # Get camera extrinsics
        R = qvec2rotmat(image.qvec)  # Rotation matrix
        t = image.tvec  # Translation vector
        
        # Get camera intrinsics
        fx = camera.params[0]  # focal length x
        fy = camera.params[1]  # focal length y
        cx = camera.params[2]  # principal point x
        cy = camera.params[3]  # principal point y
        
        # Calculate figure size based on camera dimensions
        # Convert to inches (matplotlib uses inches for figure size)
        width_inches = camera.width / 100  # Assuming 100 pixels per inch
        height_inches = camera.height / 100
        aspect_ratio = width_inches / height_inches
        
        # Create figure with aspect ratio matching camera dimensions
        plt.figure(figsize=(width_inches, height_inches))
        
        # Project points
        projected_points = []
        projected_colors = []
        for point, color in zip(all_points, pred_colors):
            # Transform point to camera coordinates
            point_cam = R @ point + t
            
            # Check if point is in front of camera
            if point_cam[2] <= 0:
                continue
                
            # Project to image plane
            x = point_cam[0] / point_cam[2]
            y = point_cam[1] / point_cam[2]
            
            # Apply camera intrinsics
            u = fx * x + cx
            v = fy * y + cy
            
            # Check if point is within image bounds
            if 0 <= u < camera.width and 0 <= v < camera.height:
                projected_points.append((u, v))
                projected_colors.append(color)
        
        if len(projected_points) > 0:
            projected_points = np.array(projected_points)
            projected_colors = np.array(projected_colors)
            
            # Plot projected points
            plt.scatter(projected_points[:, 0], projected_points[:, 1], 
                       c=projected_colors, s=1, alpha=0.5)
            
            # Set plot limits to image dimensions
            plt.xlim(0, camera.width)
            plt.ylim(camera.height, 0)  # Invert y-axis to match image coordinates
            
            # Save the visualization
            plt.axis('off')
       
            os.makedirs(os.path.join(base_dir, scan_name, 'novel_view'), exist_ok=True)
            plt.savefig(os.path.join(base_dir, scan_name, 'novel_view', image.name), 
                       bbox_inches='tight', dpi=300)
            plt.close()

    # Create video from projected and original images
    projected_dir = os.path.join(base_dir, scan_name, 'novel_view')
    original_dir = os.path.join(dataset_dir, scan_name, 'color')
    output_video = os.path.join(base_dir, scan_name, f'{scan_name}_comparison.mp4')
    create_video_from_images(projected_dir, original_dir, output_video)



def main(args):
    set_random_seeds()
    with open("edge_extraction/Replica_scans.txt", "r") as f:
        scan_names = [line.strip() for line in f]

    for scan_name in scan_names:
        process_scan(
            scan_name,
            args.base_dir,
            args.exp_name,
            args.dataset_dir,
        )



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process  data and compute metrics."
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="./output/replica/",
        help="Base directory for experiments",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="/home/gzr/data/Replica_Edge/",
        help="Directory for the dataset",
    )
    parser.add_argument("--exp_name", type=str, default="exp002", help="Experiment name")
    parser.add_argument("--downsample_density", type=float, default=0.5)
    parser.add_argument("--threshold", type=float, default=5)
    args = parser.parse_args()
    main(args)
