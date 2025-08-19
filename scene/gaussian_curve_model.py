import numpy as np
import open3d as o3d
import torch
from edge_extraction.merging import compute_pairwise_distances, compute_pairwise_cosine_similarity
from einops import rearrange
from pytorch3d.transforms import quaternion_to_matrix
from scipy.sparse.csgraph import connected_components
from simple_knn._C import distCUDA2
from torch import nn
from skimage.measure import LineModelND, ransac
from edge_extraction.fitting import line_fitting, bezier_fit
from edge_extraction.fitting import fit_straight_line
from scene.gaussian_model import GaussianModel
from utils.general_utils import get_expon_lr_func
from utils.general_utils import rot_to_quat_batch
from utils.graphics_utils import BasicPointCloud
from utils.graphics_utils import rotate_point_by_quaternion
from utils.sh_utils import RGB2SH, C0, C1, C2, C3
from utils.vis_utils import get_fancy_color

try:
    from diff_cur_rasterization import SparseGaussianAdam
except:
    pass


def initialize_bezier_curves(points, bound, n_control_points=4):
    """
    given a set of points, each point inits a curve
    B(t)=(1−t)^3P0+3(1−t)^2tP1+3(1−t)t^2P2+t^3P3
    """
    # start/end points
    assert n_control_points == 4
    # P0 = points - bound.repeat(1, 3)
    # P3 = points + bound.repeat(1, 3)
    # P1 = points - 0.5 * bound.repeat(1, 3)
    # P2 = points + 0.5 * bound.repeat(1, 3)
    direction = torch.cat([
        torch.zeros_like(bound),  # X 方向为 0
        bound,  # Y 方向为 bound
        torch.zeros_like(bound)  # Z 方向为 0
    ], dim=1)
    P0 = points - direction
    P3 = points + direction
    P1 = points - 0.5 * direction
    P2 = points + 0.5 * direction

    points_per_curve = torch.stack([P0, P1, P2, P3], dim=1)


    return points_per_curve


class GaussianCurveModel(GaussianModel):
    def __init__(self, sh_degree, n_gaussians=12, optimizer_type="default"):
        super().__init__(sh_degree, n_gaussians=12, optimizer_type=optimizer_type)
        self.n_gaussians = n_gaussians
        t = torch.linspace(0.5 / (self.n_gaussians), 1 - 0.5 / (self.n_gaussians), self.n_gaussians,
                           device='cuda')
        self.sample_t = t[:, None, None]
        self._width = torch.empty(0)
        self._mask = torch.empty(0)
        self._curve_points = torch.empty(0)
        self.is_bezier = torch.empty(0)

    @property
    def get_curve_points(self):
        return self._curve_points

    def get_curve_gaussians(self, t):
        bezier_sample_points = (1 - t) ** 3 * self._curve_points[:, 0, :] + \
            3 * (1 - t) ** 2 * t * self._curve_points[:, 1, :] \
            + 3 * (1 - t) * t ** 2 * self._curve_points[:, 2, :] + t ** 3 * self._curve_points[:, 3, :]
        if self.is_bezier.all():
            return bezier_sample_points
        line_sample_points = (1 - t) * self._curve_points[:, 0, :] + t * self._curve_points[:, 3, :]
        sample_points = torch.where(self.is_bezier.unsqueeze(0).unsqueeze(2), bezier_sample_points, line_sample_points)
        return sample_points

    def get_curve_tangent(self, t):
        bezier_sample_tangents = 3*(1-t)**2 * (self._curve_points[:, 1, :] - self._curve_points[:, 0, :]) \
            + 6*(1-t)*t * (self._curve_points[:, 2, :] - self._curve_points[:, 1, :]) \
            + 3*t**2 * (self._curve_points[:, 3, :] - self._curve_points[:, 2, :])
        if self.is_bezier.all():
            return bezier_sample_tangents
        line_sample_tangents = self._curve_points[:, 3, :] - self._curve_points[:, 0, :]
        sample_tangents = torch.where(self.is_bezier.unsqueeze(0).unsqueeze(2),
                                      bezier_sample_tangents, line_sample_tangents)
        return sample_tangents

    @property
    def get_scaling(self):
        return self._scaling

    @property
    def get_rotation_matrix(self):
        return quaternion_to_matrix(self.get_rotation)

    def get_main_axis(self, view_cam):
        rotation_matric = self.get_rotation_matrix
        dir_global = rotation_matric[..., 0]
        gaussian_to_cam_global = view_cam.camera_center - self._xyz
        neg_mask = (dir_global * gaussian_to_cam_global).sum(-1) < 0.0
        dir_global[neg_mask] = -dir_global[neg_mask]
        return dir_global

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity.unsqueeze(1).
                                       expand(-1, self.n_gaussians, -1).reshape(-1, 1))

    @property
    def get_curve_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_curve_width(self):
        return self.scaling_activation(self._width)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc.flatten(0, 1)
        features_rest = self._features_rest.flatten(0, 1)
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_features_dc(self):
        return self._features_dc.flatten(0, 1)

    @property
    def get_features_rest(self):
        return self._features_rest.flatten(0, 1)

    def create_from_pcd(self, pcd: BasicPointCloud, cam_infos: int, spatial_lr_scale: float,
                        init_size: float = 0.5, n_control_points: int = 4):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        self.dist = torch.sqrt(dist2).mean()
        bound = init_size * torch.sqrt(dist2).unsqueeze(1)
        points_per_curve = initialize_bezier_curves(fused_point_cloud, bound, n_control_points)
        opacities = self.inverse_opacity_activation(
            0.6 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        widths = self.scaling_inverse_activation(
            5e-3 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        pcd_colors = pcd.colors[:, None,:].repeat(self.n_gaussians, axis=1)
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd_colors[..., 0:1])).float().cuda())
        features = torch.zeros((pcd_colors.shape[0], pcd_colors.shape[1], 1, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :, :1, 0] = fused_color
        features[:, :, 1:, 1:] = 0.0

        self._curve_points = nn.Parameter(points_per_curve.requires_grad_(True))
        sum_n_gaussians = self._curve_points.shape[0] * self.n_gaussians
        self._features_dc = nn.Parameter(features[:, :, :, 0:1].transpose(2, 3).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, :, 1:].transpose(2, 3).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._width = nn.Parameter(widths.requires_grad_(True))
        self._mask = nn.Parameter(torch.ones((fused_point_cloud.shape[0], self.n_gaussians, 1), device="cuda")
                                  .requires_grad_(True))
        self.max_radii2D = torch.zeros(sum_n_gaussians, device="cuda")
        self.is_bezier = torch.ones(self._curve_points.shape[0], dtype=torch.bool, device='cuda')
        self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)}
        self.pretrained_exposures = None
        exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1)
        self._exposure = nn.Parameter(exposure.requires_grad_(True))
        self.prepare_scaling_rot()

    def prepare_scaling_rot(self, eps=1e-8):
        _xyz = self.get_curve_gaussians(self.sample_t)
        _xyz_front = self.get_curve_gaussians(self.sample_t-0.5/(self.n_gaussians))
        dist = torch.norm(_xyz -_xyz_front, dim=-1)
        tangent = self.get_curve_tangent(self.sample_t)
        self._xyz = rearrange(_xyz, 'm b c ->(b m) c')
        tangent = rearrange(tangent, 'm b c ->(b m) c')
        v0 = tangent / (torch.linalg.vector_norm(tangent, dim=-1, keepdim=True) + eps)
        world_up = torch.tensor([[0.0, 0.0, 1.0]], device=self._curve_points.device)
        v1 = torch.cross(tangent, world_up)
        v1 = v1 / torch.norm(v1)
        v2 = torch.cross(tangent, v1)
        v2 = v2 / torch.norm(v2)
        rotation = torch.stack((v0, v1, v2), dim=1)
        rotation = rotation.transpose(-2, -1)
        self._rotation = rot_to_quat_batch(rotation)
        s0 = rearrange(dist, 'm b ->(b m)')
        s1 = rearrange(self.get_curve_width.repeat(1, self.n_gaussians), 'b m ->(b m)')
        self._scaling = torch.stack((s0, s1, s1), dim=1)

    def training_setup(self, training_args):
            self.denom = torch.zeros((self.get_curve_points.shape[0]*self.n_gaussians, 1), device="cuda")
            self.xyz_gradient_accum = torch.zeros((self.get_curve_points.shape[0]*self.n_gaussians, 1), device="cuda")
            l = [
                {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
                {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._width], 'lr': training_args.scaling_lr, "name": "width"},
                {'params': [self._curve_points], 'lr': training_args.lr_curve_points_init, "name": "curve_points"},
                {'params': [self._mask], 'lr': training_args.mask_lr, "name": "mask"}
            ]

            if self.optimizer_type == "default":
                self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
            elif self.optimizer_type == "sparse_adam":
                try:
                    self.optimizer = SparseGaussianAdam(l, lr=0.0, eps=1e-15)
                except:
                    # A special version of the rasterizer is required to enable sparse adam
                    self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

            self.exposure_optimizer = torch.optim.Adam([self._exposure])

            self.curve_scheduler_args = get_expon_lr_func(lr_init=training_args.lr_curve_points_init,
                                                          lr_final=training_args.lr_curve_points_final,
                                                          lr_delay_mult=training_args.position_lr_delay_mult,
                                                          max_steps=training_args.position_lr_max_steps)

            self.exposure_scheduler_args = get_expon_lr_func(training_args.exposure_lr_init,
                                                             training_args.exposure_lr_final,
                                                             lr_delay_steps=training_args.exposure_lr_delay_steps,
                                                             lr_delay_mult=training_args.exposure_lr_delay_mult,
                                                             max_steps=training_args.iterations)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        if self.pretrained_exposures is None:
            for param_group in self.exposure_optimizer.param_groups:
                param_group['lr'] = self.exposure_scheduler_args(iteration)

        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "curve_points":
                lr = self.curve_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(
            torch.min(self.get_curve_opacity, torch.ones_like(self.get_curve_opacity) * 0.1))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def fix_opacity(self):
        opacities_new = self.inverse_opacity_activation(
            torch.max(self.get_curve_opacity, 0.6 *torch.ones_like(self.get_curve_opacity)))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]
        self._opacity.requires_grad = False
        for group in self.optimizer.param_groups: 
            if group['name'] == "opacity":  
                group['lr'] = 0.  



    def prune_curves(self, mask):
        valid_curves_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_curves_mask)

        self._curve_points = optimizable_tensors["curve_points"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._width = optimizable_tensors["width"]
        self._mask = optimizable_tensors["mask"]

        valid_points_mask = valid_curves_mask.unsqueeze(1).repeat(1, self.n_gaussians).flatten()
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.is_bezier = self.is_bezier[valid_curves_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        try:
            self.tmp_radii = self.tmp_radii[valid_points_mask]
        except:
            print("error in self.tmp_radii")
        self.prepare_scaling_rot()
        torch.cuda.empty_cache()

    def densification_postfix(self, new_curve_points, new_features_dc, new_features_rest,
                              new_opacities, new_widths, new_masks, new_is_bezier):

        d = {"curve_points": new_curve_points,
             "f_dc": new_features_dc,
             "f_rest": new_features_rest,
             "opacity": new_opacities,
             "width": new_widths,
             "mask": new_masks}
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._curve_points = optimizable_tensors["curve_points"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._width = optimizable_tensors["width"]
        self._mask = optimizable_tensors["mask"]
        self.is_bezier = torch.cat((self.is_bezier, new_is_bezier))
        self.xyz_gradient_accum = torch.zeros((self.get_curve_points.shape[0] * self.n_gaussians, 1), device="cuda")
        self.denom = torch.zeros((self.get_curve_points.shape[0] * self.n_gaussians, 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_curve_points.shape[0] * self.n_gaussians), device="cuda")



    def densify_and_split_curve(self, selected_pts_mask, t, N=2):
        # Extract points that satisfy the gradient condition
        new_curve_points = self.get_curve_points[selected_pts_mask].repeat(N, 1,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1, 1)
        new_opacities = self._opacity[selected_pts_mask].repeat(N, 1)
        new_widths = self._width[selected_pts_mask].repeat(N, 1)
        new_masks = self._mask[selected_pts_mask].repeat(N, 1, 1)
        new_is_bezier = self.is_bezier[selected_pts_mask].repeat(N)
        # De Casteljau’s Algorithm
        left_curves, rigth_curves = self.de_casteljau_split(self.get_curve_points[selected_pts_mask],
                                                            t, self.is_bezier[selected_pts_mask])
        new_curve_points[0:selected_pts_mask.sum(), ...] = left_curves
        new_curve_points[selected_pts_mask.sum():, ...] = rigth_curves
        self.densification_postfix(new_curve_points, new_features_dc, new_features_rest,
                                   new_opacities, new_widths, new_masks, new_is_bezier)
        prune_filter = torch.cat(
            (selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_curves(prune_filter)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        self.tmp_radii = radii
        grads = rearrange(grads, '(b m) c-> b m c', m=self.n_gaussians)
        max_values, max_indices = torch.max(torch.norm(grads, dim=-1) , dim=1)
        selected_pts_mask = max_values >= max_grad
        if selected_pts_mask.sum()>0:
            j_idx = max_indices[selected_pts_mask]
            t = self.sample_t[j_idx]
            self.densify_and_split_curve(selected_pts_mask, t.squeeze(-1))

        prune_mask = (self.get_curve_opacity < min_opacity).squeeze()
        self.prune_curves(prune_mask)
        torch.cuda.empty_cache()


    def de_casteljau_trim(self, curves, from_t ,end_t, is_bezier):
        _, right_curves = self.de_casteljau_split(curves, from_t, is_bezier)
        left_curves, _ = self.de_casteljau_split(right_curves, end_t, is_bezier)
        return left_curves

    def curve_split_curvature(self, threshold_angle=20, threshold_radian_skip=30):
        threshold_radian = torch.tensor(threshold_angle * (torch.pi / 180))
        threshold_radian_skip = torch.tensor(threshold_radian_skip * (torch.pi / 180))
        curvature = rearrange(self.get_rotation_matrix[..., 0], '(b m) c-> b m c', m=self.n_gaussians) # normalized
        cos_theta = torch.einsum('bij,bij->bi', curvature[:, :-1, :], curvature[:, 1:, :])
        angles = torch.acos(cos_theta.clamp(-1, 1))
        cos_theta_skip = torch.einsum('bij,bij->bi', curvature[:, :-2, :], curvature[:, 2:, :])
        angles_skip = torch.acos(cos_theta_skip.clamp(-1, 1))
        mask_split = torch.max(angles, dim=-1).values > threshold_radian
        mask_skip = torch.max(angles_skip, dim=-1).values > threshold_radian_skip
        mask_split |= mask_skip
        angles_max, t = torch.max(angles, dim=-1)
        end_t = self.sample_t[t] + 0.5 / self.n_gaussians
        self.densify_and_split_curve(mask_split, end_t[mask_split].squeeze(-1))
        torch.cuda.empty_cache()
        self.prepare_scaling_rot()

    def de_casteljau_split(self, curves, t, is_bezier):
        Q0 = (1 - t) * curves[:, 0, :] + t * curves[:, 1, :]
        Q1 = (1 - t) * curves[:, 1, :] + t * curves[:, 2, :]
        Q2 = (1 - t) * curves[:, 2, :] + t * curves[:, 3, :]
        R0 = (1 - t) * Q0 + t * Q1
        R1 = (1 - t) * Q1 + t * Q2
        S = (1 - t) * R0 + t * R1
        left_bezier = torch.stack([curves[:, 0],
                                   Q0,
                                   R0,
                                   S], dim=1)

        right_bezier = torch.stack([S,
                                    R1,
                                    Q2,
                                    curves[:, -1]], dim=1)
        if self.is_bezier.all():
            return left_bezier, right_bezier
        S = (1 - t) * curves[:, 0] + t * curves[:, -1]  # [B, 3]
        left_straight = torch.stack([
            curves[:, 0],
            (2 / 3) * curves[:, 0] + (1 / 3) * S,
            (1 / 3) * curves[:, 0] + (2 / 3) * S,
            S
        ], dim=1)  # [B, 4, 3]
        right_straight = torch.stack([
            S,
            (2 / 3) * S + (1 / 3) * curves[:, -1],
            (1 / 3) * S + (2 / 3) * curves[:, -1],
            curves[:, -1]
        ], dim=1)
        left = torch.where(is_bezier[:, None, None], left_bezier, left_straight)  # [B, 4, 3]
        right = torch.where(is_bezier[:, None, None], right_bezier, right_straight)
        return left, right


    def only_prune(self, min_opacity, mask_threshold):
        prune_mask = torch.logical_or((torch.sigmoid(self._mask) <= mask_threshold).all(dim=1).squeeze(),
                                      (self.get_curve_opacity < min_opacity).squeeze())
       
        small_mask = self._scaling[:, 0].clone().detach().reshape(-1, self.n_gaussians).sum(-1) < 1e-2
        prune_mask = torch.logical_or(small_mask, prune_mask)
        self.prune_curves(prune_mask)
        torch.cuda.empty_cache()

    def mask_trim_split(self, mask_threshold):
        valid_mask = (torch.sigmoid(self._mask) > mask_threshold).squeeze()
        # trim and split curves
        # step1 : Trim both ends of the curve
        start_idx = torch.argmax(valid_mask.int(), dim=1)
        reversed_mask = torch.flip(valid_mask, [1])
        end_idx = self.n_gaussians -1 - torch.argmax(reversed_mask.int(), dim=1)
        from_t = self.sample_t[start_idx, :, :].squeeze(-1)
        end_t = self.sample_t[end_idx, :, :].squeeze(-1)
        from_t = from_t - 0.5 / self.n_gaussians
        end_t = end_t + 0.5 / self.n_gaussians
        trim_curve_points = self.de_casteljau_trim(self.get_curve_points, from_t, end_t, self.is_bezier)
        trim_curve_mask = self._mask.clone().detach()
        mask = (start_idx != 0) | (end_idx != self.n_gaussians -1)
        for i in torch.nonzero(mask).squeeze(-1):
            _mask_i =trim_curve_mask[i][start_idx[i]:end_idx[i]+1]
            inter_mask=torch.nn.functional.interpolate(_mask_i.unsqueeze(0).unsqueeze(0), 
                                                       size=(self.n_gaussians,1), mode='bilinear')
            trim_curve_mask[i] = inter_mask.unsqueeze(0).unsqueeze(0)
        optimizable_tensors = self.replace_tensor_to_optimizer(trim_curve_mask, 'mask')
        self._mask = optimizable_tensors["mask"]
        optimizable_tensors = self.replace_tensor_to_optimizer(trim_curve_points, 'curve_points')
        self._curve_points = optimizable_tensors["curve_points"]
        self.prepare_scaling_rot()


    def merge_curves(self, distance_threshold=0.02, similarity_threshold=0.97, sample_num=100,
                     ransac_thresh=0.005):
        # connect start/end conncet
        with torch.no_grad():
            # bezier curve merger
            t = torch.linspace(0, 1, sample_num, device='cuda')
            t = t[:, None, None]
            curve_sample_points = rearrange(self.get_curve_gaussians(t), 'm b c -> b m c')
            num_curves = self.get_curve_points.shape[0]
            curve_points = self.get_curve_points
            start_points, end_points = curve_points[:, 0], curve_points[:, -1]
            all_points = torch.cat([start_points, end_points], dim=0)
            start_tangs = curve_points[:, 1] - curve_points[:, 0]
            end_tangs = curve_points[:, 2] - curve_points[:, -1]
            all_tangs = torch.cat([start_tangs, end_tangs], dim=0)
            all_tangs = all_tangs / (torch.norm(all_tangs, dim=-1, keepdim=True)+1e-6)
            similarity = torch.abs(torch.sum(all_tangs[None,:,:] * all_tangs[:,None,:], dim=-1))
            dist = torch.cdist(all_points, all_points, p=2)
            mask_merge = (dist < 2*distance_threshold) & (similarity> similarity_threshold)
            adjacency_matrix = (mask_merge[0:num_curves, 0:num_curves] |
                            mask_merge[0:num_curves, num_curves:] |
                            mask_merge[num_curves:, 0:num_curves] |
                            mask_merge[num_curves:, num_curves:])
            confidence_matrix = torch.max(torch.max(similarity[0:num_curves, 0:num_curves],  similarity[0:num_curves, num_curves:]),
                                          torch.max(similarity[num_curves:, 0:num_curves], similarity[num_curves:, num_curves:]))
            # num_components, labels = connected_components(adjacency_matrix.cpu().numpy(), directed=True)
            merge_mask = torch.zeros(num_curves, dtype=torch.bool, device='cuda')
            new_curve_points = []
            new_features_dc = []
            new_features_rest = []
            new_opacities = []
            new_widths = []
            new_masks = []
            new_is_bezier = []
            components_indices = []
            merged = set()
            for i in range(num_curves):
                if (i in merged) or (self.is_bezier[i].item() == False): #  w/o item note that is false is wrong
                    continue
                neighbors = torch.nonzero(adjacency_matrix[i]).squeeze(1).tolist()

                neighbors = [j for j in neighbors if j not in merged and j != i and self.is_bezier[j].item()]
                if not neighbors:
                    continue  
                best_j = max(neighbors, key=lambda j: confidence_matrix[i, j])
                merged.add(i)
                merged.add(best_j)
                components_indices.append([i, best_j])

            for component_indices in components_indices:

                raw_points_group = [curve_sample_points[i] for i in component_indices]
                pts_curr = torch.cat(raw_points_group, dim=0).cpu().numpy()
                try:
                    _, inliers = ransac(pts_curr, LineModelND, min_samples=2,
                                        residual_threshold=ransac_thresh, max_trials=1000)
                    line = pts_curr[inliers]
                    line_eps, _ = line_fitting(line)
                except:
                    continue
                main_direction = line_eps[3:] - line_eps[:3]
                main_direction /= np.linalg.norm(main_direction)
                mean_pt = (line_eps[3:] + line_eps[:3]) / 2

                lines_to_point = (pts_curr - mean_pt)
                dirs_to_point = lines_to_point / np.linalg.norm(lines_to_point, axis=1)[:, np.newaxis]
                normals = np.cross(main_direction, dirs_to_point)
                normals /= np.linalg.norm(normals, axis=1)[:, np.newaxis]
                normals2 = np.cross(main_direction, normals)
                normals2 /= np.linalg.norm(normals2, axis=1)[:, np.newaxis]
                lamdas = np.dot(lines_to_point, main_direction)
                lamda_order = np.argsort(lamdas)
                pts_curr = pts_curr[lamda_order]

                # now fit a bezier curve through the points and see if the residuals change quite a bit
                # if they do, then this is probably a curve
                out = bezier_fit(pts_curr, error_threshold=distance_threshold)
                if out is not None:
                    # from edge_extraction.fitting import bezier_curve
                    # sample_points = bezier_curve(t.squeeze().cpu().numpy(), *out).reshape(-1, 3)
                    merge_mask[component_indices] = True
                    new_curve_points.append(torch.from_numpy(out.reshape(-1, 3)).float().cuda().unsqueeze(0))
                    new_features_dc.append(self._features_dc[0:1])
                    new_features_rest.append(self._features_rest[0:1])
                    new_is_bezier.append(torch.ones_like(self.is_bezier[0:1], dtype=torch.bool))
                    new_opacities.append(self._opacity[component_indices].mean(dim=0,keepdim=True))
                    new_widths.append(self._width[component_indices].mean(dim=0, keepdim=True))
                    new_masks.append(torch.ones_like(self._mask[0:1]))

        # merge lines
        line_idx = torch.where(self.is_bezier == False)[0]
        if len(line_idx)>0:
            with torch.no_grad():
                line_segments = rearrange(self._curve_points[line_idx][:, [0,-1], :],
                                          'b m c -> b (m c)').cpu().numpy()

                dist_matrix = compute_pairwise_distances(line_segments)
                similarity_matrix = compute_pairwise_cosine_similarity(line_segments)
                similarity_matrix = np.abs(similarity_matrix)
                # Create adjacency matrix based on distance and similarity thresholds
                adjacency_matrix = (dist_matrix <= distance_threshold) & (
                        similarity_matrix >= similarity_threshold
                )
                # Compute connected components
                num_components, labels = connected_components(adjacency_matrix)
                for component in range(num_components):
                    component_indices = np.where(labels == component)[0]
                    if len(component_indices) == 1:
                        continue
                    else:
                        component_indices = line_idx[component_indices]
                        merge_mask[component_indices] = True
                        pts_curr = curve_sample_points[component_indices].cpu().numpy().reshape(-1, 3)

                        start, end, direction, mean_point, t_min, t_max = \
                            fit_straight_line(pts_curr)
                        out = np.zeros((4,3), dtype=np.float32)
                        out[0, :] = start
                        out[-1, :] = end
                        new_curve_points.append(torch.from_numpy(out).float().cuda().unsqueeze(0))
                        new_features_dc.append(self._features_dc[0:1])
                        new_features_rest.append(self._features_rest[0:1])
                        new_is_bezier.append(torch.zeros_like(self.is_bezier[0:1], dtype=torch.bool))
                        new_opacities.append(self._opacity[component_indices].mean(dim=0, keepdim=True))
                        new_widths.append(self._width[component_indices].mean(dim=0, keepdim=True))
                        new_masks.append(torch.ones_like(self._mask[0:1]))


        if merge_mask.any():
            self.prune_curves(merge_mask)

            self.densification_postfix(torch.cat(new_curve_points), torch.cat(new_features_dc), torch.cat(new_features_rest),
                                        torch.cat(new_opacities), torch.cat(new_widths),torch.cat(new_masks), torch.cat(new_is_bezier))
            self.prepare_scaling_rot()

    def fit_curve_to_line(self, threshold=0.002, threshold_max=0.004, sample_num=100):
        t = torch.linspace(0, 1, sample_num, device='cuda')
        t = t[:, None, None]
        curve_sample_points = rearrange(self.get_curve_gaussians(t), 'm b c -> b m c')
        lines_points = []
        num_curves = self.get_curve_points.shape[0]
        selected_mask = torch.zeros(num_curves, dtype=torch.bool, device='cuda')
        for i in range(num_curves):
            if self.is_bezier[i].item() is False:
                continue
            is_line, start, end = self.is_curve_straight(curve_sample_points[i],
                                                         threshold=threshold,
                                                         threshold_max=threshold_max)
            if is_line:
                selected_mask[i] = True
                lines_points.append(np.stack([start, end]))
        if selected_mask.any():
            # delete the transformed curves
            lines_points = torch.tensor(np.array(lines_points)).cuda()
            self.is_bezier[selected_mask] = False
            new_curve_points = self._curve_points.clone().detach()
            new_curve_points[selected_mask][:, 0] = lines_points[:, 0, :]
            new_curve_points[selected_mask][:, -1] = lines_points[:, -1, :]
            optimizable_tensors = self.replace_tensor_to_optimizer(new_curve_points, 'curve_points')
            self._curve_points = optimizable_tensors["curve_points"]
            print('number of bezier/curves:', self.is_bezier.sum(), self._curve_points.shape[0])



    def is_curve_straight(self, sample_points, threshold=0.002, threshold_max=0.004):
        sample_points = sample_points.cpu().numpy()
        start, end, direction, mean_point, t_min, t_max = fit_straight_line(sample_points)
        vectors_to_points = sample_points - mean_point
        t = np.dot(vectors_to_points, direction)
        closest_points = mean_point + np.clip(t, t_min, t_max).reshape(-1, 1) * direction
        distances = np.linalg.norm(sample_points - closest_points, axis=1)
        return (np.mean(distances) < threshold) & (distances.max() < threshold_max), start, end


    @torch.no_grad()
    def draw_ellipsoids(self, path, step, radius=1.2):
        xyzs = (self.get_xyz)  # [gs_num, 3]
        rotations = (self.get_rotation)  # [gs_num, 4]
        scales = (self.get_scaling)  # [gs_num, 3]
        features = (self.get_features)  # [gs_num, 3, 16]
        p1 = rotate_point_by_quaternion(rotations[:, None, :][:, 0, :], torch.tensor([[0, 0, 1]]).float().cuda())
        p2 = rotate_point_by_quaternion(rotations[:, None, :][:, 0, :], torch.tensor([[0, 0, 0]]).float().cuda())
        dir = p1 - p2
        colors = C0 * features[:, 0, :]  # [gs_num, 3]
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        if (self.active_sh_degree > 0):
            x = dir[:, 0:1]
            y = dir[:, 1:2]
            z = dir[:, 2:3]
            colors = colors - C1 * y * features[:, 1, :] + C1 * z * features[:, 2, :] - C1 * x * features[:, 3, :]
            if (self.active_sh_degree > 1):
                xx = x * x
                yy = y * y
                zz = z * z
                xy = x * y
                yz = y * z
                xz = x * z
                colors = colors + \
                         C2[0] * xy * features[:, 4, :] + \
                         C2[1] * yz * features[:, 5, :] + \
                         C2[2] * (2 * zz - xx - yy) * features[:, 6, :] + \
                         C2[3] * xz * features[:, 7, :] + \
                         C2[4] * (xx - yy) * features[:, 8, :]

                if (self.active_sh_degree > 2):
                    colors = colors + \
                             C3[0] * y * (3 * xx - yy) * features[:, 9, :] + \
                             C3[1] * xy * z * features[:, 10, :] + \
                             C3[2] * y * (4 * zz - xx - yy) * features[:, 11, :] + \
                             C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * features[:, 12, :] + \
                             C3[4] * x * (4 * zz - xx - yy) * features[:, 13, :] + \
                             C3[5] * z * (xx - yy) * features[:, 14, :] + \
                             C3[6] * x * (xx - 3 * yy) * features[:, 15, :]
        n_curves = self.get_curve_points.shape[0]
        colors = get_fancy_color(n_curves+1)
        colors = colors[torch.randperm(n_curves)]
        colors = colors[:, None, :].repeat(1, self.n_gaussians, 1).view(-1, 3)
        lines_mask = ~self.is_bezier
        colors[lines_mask[:, None].repeat(1, self.n_gaussians).view(-1, 1)[:, 0]] = 0.
        _mask_mask = (torch.sigmoid(self._mask) < 0.01).view(-1, 1)
        colors[_mask_mask[:, 0]] = 1.

        def generate_ellipse_mesh(center, rotation, scale, color):
            mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=10)
            vertices = np.asarray(mesh.vertices)
            scale_matrix = np.diag([scale[0], scale[1], scale[2]])  
            vertices = vertices @ scale_matrix.T
            mesh.vertices = o3d.utility.Vector3dVector(vertices)
            rotation_matrix = o3d.geometry.get_rotation_matrix_from_quaternion(rotation)
            mesh.rotate(rotation_matrix, center=(0, 0, 0))
            mesh.translate(center)
            mesh.paint_uniform_color(np.clip(color, 0, 1))

            return mesh

        meshes = []
        for i in range(len(xyzs)):
            center = xyzs[i].cpu().numpy()  
            rotation = rotations[i].cpu().numpy()  
            scale = [scales[i, 0].cpu().numpy(), scales[i, 1].cpu().numpy(), scales[i, 2].cpu().numpy()]
            color = colors[i].cpu().numpy()  
            mesh = generate_ellipse_mesh(center, rotation, scale, color)
            meshes.append(mesh)

        combined_mesh = o3d.geometry.TriangleMesh()
        for mesh in meshes:
            combined_mesh += mesh
        output_path = f"{path}/ellipsoids_step{step}.ply"
        o3d.io.write_triangle_mesh(output_path, combined_mesh)
        print(f"Saved mesh to {output_path}")


    @torch.no_grad()
    def draw_curve(self, path, step, num_sample=200):
        n_curves = self.get_curve_points.shape[0]
        colors = get_fancy_color(n_curves+1)
        colors = colors[torch.randperm(n_curves)]
        colors = colors[:, None, :].repeat(1, num_sample, 1).view(-1, 3)

        sample_t = torch.linspace(0, 1, num_sample, device='cuda')
        sample_t = sample_t[:, None, None]
        sampled_points = rearrange(self.get_curve_gaussians(sample_t), 'm b c -> b m c')
        sampled_points = sampled_points.reshape(-1, 3)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(sampled_points.cpu().numpy())
        pcd.colors = o3d.utility.Vector3dVector(colors.cpu().numpy())
        output_path = f"{path}/curve_step{step}.ply"
        o3d.io.write_point_cloud(output_path, pcd, write_ascii=True)
