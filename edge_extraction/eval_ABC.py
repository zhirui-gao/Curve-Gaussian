import os
import numpy as np
import argparse
from numpy.linalg import norm
import sys
from datetime import datetime
import logging, os, sys
from datetime import datetime
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from edge_extraction.eval_utils import (
    set_random_seeds,
    compute_chamfer_distance,
    compute_precision_recall_IOU,
    downsample_point_cloud_average,
    get_gt_points,
    get_pred_points_and_directions,
    visualize_pred_gt
)
from scipy.spatial import cKDTree
import open3d as o3d
import matplotlib.pyplot as plt
from scene.dataset_readers import readCamerasFromTransforms

def compute_direction_similarity(
    pred_points, pred_directions, gt_points, gt_directions
):
    # Create a KDTree for efficient nearest neighbor search
    tree = cKDTree(gt_points)
    distances, indices = tree.query(pred_points, k=1)
    similarities = []
    for idx, pred_dir in zip(indices, pred_directions):
        gt_dir = gt_directions[idx]
        cosine_similarity = np.dot(pred_dir, gt_dir) / (norm(pred_dir) * norm(gt_dir))
        similarities.append(np.abs(cosine_similarity))
    return np.array(similarities).mean()


def update_totals_and_metrics(metrics, totals, results, edge_type):
    correct_gt, num_gt, correct_pred, num_pred, acc, comp = results
    metrics[f"comp_{edge_type}"].append(comp)
    metrics[f"acc_{edge_type}"].append(acc)
    for i, threshold in enumerate(["5", "10", "20"]):
        totals[f"thre{threshold}_correct_gt_total"] += correct_gt[i]
        totals[f"thre{threshold}_correct_pred_total"] += correct_pred[i]
    totals["num_gt_total"] += num_gt
    totals["num_pred_total"] += num_pred


def finalize_metrics(metrics):
    for key, value in metrics.items():
        value = np.array(value)
        value[np.isnan(value)] = 0
        metrics[key] = round(np.mean(value), 4)
    return metrics


def print_metrics(metrics, edge_type):
    logging.info(f"{edge_type.capitalize()}:")
    logging.info(f"  Completeness: {metrics[f'comp_{edge_type}']}")
    logging.info(f"  Accuracy: {metrics[f'acc_{edge_type}']}")


def project_points_to_camera(points, colors, camera_info):
    """
    Project 3D points onto camera view using camera parameters.
    
    Args:
        points: Nx3 array of 3D points
        colors: Nx3 array of point colors
        camera_info: CameraInfo object containing camera parameters
    """
    # Get camera parameters
    R = np.transpose(camera_info.R)  # Rotation matrix
    T = camera_info.T  # Translation vector
    FovX = camera_info.FovX  # Field of view in x direction
    FovY = camera_info.FovY  # Field of view in y direction
    width = camera_info.width
    height = camera_info.height
    # Calculate focal lengths from FOV
    fx = width / (2 * np.tan(FovX / 2))
    fy = height / (2 * np.tan(FovY / 2))
    cx = width / 2
    cy = height / 2
    # Project points
    projected_points = []
    projected_colors = []
    
    for point, color in zip(points, colors):
        # Transform point to camera coordinates
        point_cam = R @ point + T
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
        if 0 <= u < width and 0 <= v < height:
            projected_points.append((u, v))
            projected_colors.append(color)
    
    return np.array(projected_points), np.array(projected_colors)

def visualize_projection(points, colors, camera_info, save_path):
    """
    Visualize projected points on camera view.
    
    Args:
        points: Nx3 array of 3D points
        colors: Nx3 array of point colors
        camera_info: CameraInfo object containing camera parameters
        save_path: Path to save the visualization
    """
    # Project points
    projected_points, projected_colors = project_points_to_camera(points, colors, camera_info)
    if len(projected_points) > 0:
        # Calculate figure size based on camera dimensions
        width_inches = camera_info.width / 200  # Assuming 100 pixels per inch
        height_inches = camera_info.height / 200
        # Create figure
        plt.figure(figsize=(width_inches, height_inches))
        # Plot projected points
        plt.scatter(projected_points[:, 0], projected_points[:, 1], 
                   c=projected_colors, s=1, alpha=0.5)
        # Set plot limits to image dimensions
        plt.xlim(0, camera_info.width)
        plt.ylim(camera_info.height, 0)  # Invert y-axis to match image coordinates
        # Save the visualization
        plt.axis('off')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()

def process_scan(scan_name, base_dir, dataset_dir, metrics, totals, render_mv):
    logging.info(f"Processing: {scan_name}")
    json_path = os.path.join(
        base_dir, scan_name, "parametric_edges.json"
    )
    if not os.path.exists(json_path):
        logging.info(f"Invalid prediction at {scan_name}")
        return
    all_curve_points, all_line_points, all_curve_directions,\
        all_line_directions, all_curve_colors, all_line_colors,\
        num_curves, num_lines= (
        get_pred_points_and_directions(json_path)
    )
    all_curve_directions = np.array(all_curve_directions)
    all_line_directions = np.array(all_line_directions)
    if  all_curve_directions.shape[0] == 0:
        all_curve_directions = np.zeros((0, 3), dtype=np.float32)  # 初始化为空的 2 维数组
    else:
        all_curve_directions = np.asarray(all_curve_directions).reshape(-1, 3).astype(np.float32)

    if all_line_directions.shape[0] == 0:
        all_line_directions = np.zeros((0, 3), dtype=np.float32)  # 初始化为空的 2 维数组
    else:
        all_line_directions = np.asarray(all_line_directions).reshape(-1, 3).astype(np.float32)
    pred_directions = (
        np.concatenate([all_curve_directions, all_line_directions], axis=0)
    )
    pred_points = (
        np.concatenate([all_curve_points, all_line_points], axis=0)
        .reshape(-1, 3)
        .astype(np.float32)
    )
    pred_colors =  (
        np.concatenate([all_curve_colors, all_line_colors], axis=0)
        .reshape(-1, 3)
        .astype(np.float32)
    )
    if len(pred_points) == 0:
        logging.info(f"Invalid prediction at {scan_name}")
        return
    if render_mv:
        test_cam_infos = readCamerasFromTransforms(os.path.join(dataset_dir,  'data', scan_name),'transforms_video.json')
        # Project points to each camera view
        for i, cam_info in enumerate(test_cam_infos):
            save_path = os.path.join(base_dir, scan_name, 'novel_view', cam_info.image_name)
            visualize_projection(pred_points, pred_colors, cam_info, save_path)
    else:
        pred_sampled = downsample_point_cloud_average(
            pred_points,
            num_voxels_per_axis=256,
            min_bound=[0, 0, 0],
            max_bound=[1, 1, 1],
        )

        gt_points_raw, gt_points, gt_points_direction, gt_points_color = get_gt_points(
            scan_name, "all", data_base_dir=os.path.join(dataset_dir, "groundtruth"), return_direction=True
        )
        if gt_points_raw is None:
            return

        similarity = compute_direction_similarity(
            pred_points, pred_directions, gt_points, gt_points_direction
        )

        chamfer_dist, acc, comp = compute_chamfer_distance(pred_sampled, gt_points)
        visualize_pred_gt(pred_points, gt_points, pred_colors,
                        scan_name, save_fig=True, show_fig=False, vis_dir=os.path.dirname(json_path),
                        gt_points_color=gt_points_color)
        logging.info(
        f"  Chamfer Distance: {chamfer_dist:.4f}, Accuracy: {acc:.4f}, Completeness: {comp:.4f}    Norm: {similarity:.4f}"
        )
        metrics["chamfer"].append(chamfer_dist)
        metrics["acc"].append(acc)
        metrics['simi'].append(similarity)
        metrics["comp"].append(comp)
        metrics["num_curves"].append(num_lines)
        metrics["num_lines"].append(num_curves)
        metrics = compute_precision_recall_IOU(
            pred_sampled,
            gt_points,
            metrics,
            thresh_list=[0.005, 0.01, 0.02],
            edge_type="all",
        )

        for edge_type in ["curve", "line"]:
            gt_points_raw_edge, gt_points_edge, _, _ = get_gt_points(
                scan_name,
                edge_type,
                return_direction=True,
                data_base_dir=os.path.join(dataset_dir, "groundtruth"),
            )
            if gt_points_raw_edge is not None:
                results = compute_precision_recall_IOU(
                    pred_sampled,
                    gt_points_edge,
                    None,
                    thresh_list=[0.005, 0.01, 0.02],
                    edge_type=edge_type,
                )
                update_totals_and_metrics(metrics, totals[edge_type], results, edge_type)



def main(base_dir, dataset_dir, render_mv):
    set_random_seeds()
    metrics = {
        "chamfer": [],
        "acc": [],
        'simi':[],
        "num_lines": [],
        "num_curves": [],
        "comp": [],
        "comp_curve": [],
        "comp_line": [],
        "acc_curve": [],
        "acc_line": [],
        "precision_0.01": [],
        "recall_0.01": [],
        "fscore_0.01": [],
        "IOU_0.01": [],
        "precision_0.02": [],
        "recall_0.02": [],
        "fscore_0.02": [],
        "IOU_0.02": [],
        "precision_0.005": [],
        "recall_0.005": [],
        "fscore_0.005": [],
        "IOU_0.005": [],
    }

    totals = {
        "curve": {
            "thre5_correct_gt_total": 0,
            "thre10_correct_gt_total": 0,
            "thre20_correct_gt_total": 0,
            "thre5_correct_pred_total": 0,
            "thre10_correct_pred_total": 0,
            "thre20_correct_pred_total": 0,
            "num_gt_total": 0,
            "num_pred_total": 0,
        },
        "line": {
            "thre5_correct_gt_total": 0,
            "thre10_correct_gt_total": 0,
            "thre20_correct_gt_total": 0,
            "thre5_correct_pred_total": 0,
            "thre10_correct_pred_total": 0,
            "thre20_correct_pred_total": 0,
            "num_gt_total": 0,
            "num_pred_total": 0,
        },
    }

    # with open("edge_extraction/ABC_scans.txt", "r") as f:
    #     scan_names = [line.strip() for line in f]

    scan_names = sorted([f.name for f in os.scandir(dataset_dir) if f.is_dir()])

    for scan_name in scan_names:
        process_scan(scan_name, base_dir, os.path.dirname(dataset_dir), metrics, totals, render_mv)

    metrics = finalize_metrics(metrics)

    logging.info("Summary:")
    logging.info(f"  Number line/curve: {metrics['num_lines']}, {metrics['num_curves']}")
    logging.info(f"  Accuracy: {metrics['acc']:.4f}")
    logging.info(f"  Completeness: {metrics['comp']:.4f}")
    logging.info(f"  Norm: {metrics['simi']:.4f}")
    logging.info(f"  Recall @ 5 mm: {metrics['recall_0.005']:.4f}")
    logging.info(f"  Recall @ 10 mm: {metrics['recall_0.01']:.4f}")
    logging.info(f"  Recall @ 20 mm: {metrics['recall_0.02']:.4f}")
    logging.info(f"  Precision @ 5 mm: {metrics['precision_0.005']:.4f}")
    logging.info(f"  Precision @ 10 mm: {metrics['precision_0.01']:.4f}")
    logging.info(f"  Precision @ 20 mm: {metrics['precision_0.02']:.4f}")
    logging.info(f"  F-Score @ 5 mm: {metrics['fscore_0.005']:.4f}")
    logging.info(f"  F-Score @ 10 mm: {metrics['fscore_0.01']:.4f}")
    logging.info(f"  F-Score @ 20 mm: {metrics['fscore_0.02']:.4f}")

    if totals["curve"]["num_gt_total"] > 0:
        print_metrics(metrics, "curve")
    else:
        logging.info("Curve: No ground truth edges found.")

    if totals["line"]["num_gt_total"] > 0:
        print_metrics(metrics, "line")
    else:
        logging.info("Line: No ground truth edges found.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process CAD data and compute metrics."
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="./output",
        help="Base directory for experiments",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="/media/gzr/955be20b-af2b-4597-83f8-8585ff878672/ABC_dataset/ABC-NEF_Edge/data",  # 
        help="Directory for the dataset",
    )
    parser.add_argument("--exp_name", type=str, default="exp01", help="Experiment name")
    parser.add_argument("--render_mv", type=bool, default=False, help="Render mv")
    args = parser.parse_args()
   
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(args.base_dir, f"output_{timestamp}.txt")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info("start to evaluate……")
    main(args.base_dir, args.dataset_dir, args.render_mv)
    logging.info("evaluate file save to %s", log_file)

 