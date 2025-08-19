import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score
from scipy.sparse.csgraph import connected_components
import open3d as o3d
from scipy.spatial.distance import euclidean, cdist
import os


def merge_endpoints(merged_line_segments, merged_bezier_curves, distance_threshold):
    N_lines = len(merged_line_segments)
    N_curves = len(merged_bezier_curves)

    if N_lines == 0 and N_curves == 0:
        return [], []

    if N_lines > 0:
        line_endpoints = merged_line_segments.reshape(-1, 3)
    else:
        line_endpoints = np.array([]).reshape(-1, 3)

    if N_curves > 0:
        curve_endpoints = merged_bezier_curves[:, [0, 1, 2, -3, -2, -1]].reshape(-1, 3)
    else:
        curve_endpoints = np.array([]).reshape(-1, 3)

    concat_endpoints = np.concatenate([line_endpoints, curve_endpoints], axis=0)

    dist_matrix = cdist(concat_endpoints, concat_endpoints)
    adjacency_matrix = dist_matrix <= distance_threshold
    num_components, labels = connected_components(adjacency_matrix)
    for component in range(num_components):
        component_indices = np.where(labels == component)[0]
        if len(component_indices) > 1:
            endpoints = concat_endpoints[component_indices]
            mean_endpoint = np.mean(endpoints, axis=0)
            concat_endpoints[component_indices] = mean_endpoint

    if N_lines > 0:
        merged_line_segments_merged_endpoints = concat_endpoints[: N_lines * 2].reshape(
            -1, 6
        )
    else:
        merged_line_segments_merged_endpoints = []

    if N_curves > 0:
        merged_curve_segments_merged_endpoints = np.zeros_like(merged_bezier_curves)
        curve_merged_endpoints = concat_endpoints[N_lines * 2 :].reshape(-1, 6)
        merged_curve_segments_merged_endpoints[:, :3] = curve_merged_endpoints[:, :3]
        merged_curve_segments_merged_endpoints[:, 3:9] = merged_bezier_curves[:, 3:9]
        merged_curve_segments_merged_endpoints[:, 9:] = curve_merged_endpoints[:, 3:]

    else:
        merged_curve_segments_merged_endpoints = []

    return merged_line_segments_merged_endpoints, merged_curve_segments_merged_endpoints

def compute_pairwise_cosine_similarity(line_segments):
    direction_vectors = line_segments[:, 3:] - line_segments[:, :3]
    pairwise_similarity = cosine_similarity(direction_vectors)
    return pairwise_similarity

def line_segment_point_distance(line_segment, query_point):
    """Compute the Euclidean distance between a line segment and a query point.

    Parameters:
        line_segment (np.ndarray): An array of shape (6,), representing two 3D endpoints.
        query_point (np.ndarray): An array of shape (3,), representing the 3D query point.

    Returns:
        float: The minimum distance from the query point to the line segment.
    """
    point1, point2 = line_segment[:3], line_segment[3:]
    point_delta = point2 - point1
    u = np.clip(
        np.dot(query_point - point1, point_delta) / np.dot(point_delta, point_delta),
        0,
        1,
    )
    closest_point = point1 + u * point_delta
    return np.linalg.norm(closest_point - query_point)


def compute_pairwise_distances(line_segments):
    """Compute pairwise distances between line segments.

    Parameters:
        line_segments (np.ndarray): An array of shape (N, 6), each row represents a line segment in 3D.

    Returns:
        np.ndarray: A symmetric array of shape (N, N), containing pairwise distances.
    """
    num_lines = len(line_segments)
    endpoints = line_segments.reshape(-1, 3)
    dist_matrix = np.zeros((num_lines, num_lines))

    for i, line_segment in enumerate(line_segments):
        for j in range(i + 1, num_lines):
            min_distance = min(
                line_segment_point_distance(line_segment, endpoints[2 * j]),
                line_segment_point_distance(line_segment, endpoints[2 * j + 1]),
            )
            dist_matrix[i, j] = min_distance

    dist_matrix += dist_matrix.T  # Make the matrix symmetric
    return dist_matrix
