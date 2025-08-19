import torch.nn.functional as F
import os
import open3d as o3d
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim,  edge_aware_loss
from gaussian_renderer import render
import sys
from edge_extraction.merging import merge_endpoints
from scene import Scene, GaussianCurveModel
from utils.general_utils import safe_state
import uuid
import torch.nn.functional as F
import json
from tqdm import tqdm
from einops import rearrange
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, OptimizationParamsPidinet, OptimizationParamsReplica
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_cur_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, 
             checkpoint_iterations, checkpoint, debug_from):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianCurveModel(dataset.sh_degree, dataset.n_gaussians, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    bg = torch.rand((3), device="cuda") if opt.random_background else background
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_normal_for_log = 0.0
    ema_product_for_log = 0.0
    curve_smo_for_log = 0.0
    curve_conn_for_log = 0.0
    normal_prior_error = torch.tensor(0.0).cuda()
    dot_products = torch.tensor(0.0).cuda()
    curve_smo = torch.tensor(0.0).cuda()
    curve_conn= torch.tensor(0.0).cuda()
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    reset_timestep = 0
    for iteration in range(first_iter, opt.iterations + 1):
        reset_timestep += 1
        iter_start.record()
        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp,
                            separate_sh=SPARSE_ADAM_AVAILABLE,
                            use_mask=iteration>=opt.densify_until_iter, mask_thr=opt.mask_threshold)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], \
            render_pkg["visibility_filter"], render_pkg["radii"]
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = edge_aware_loss(image, gt_image[:1, ...])
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image[:1, ...].unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image[:1, ...])

        loss = opt.lambda_mse * ((1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value))

        #regularization
        if iteration>=opt.densify_until_iter:
            loss = loss + opt.lambda_mask * torch.mean((torch.sigmoid(gaussians._mask)))


        if visibility_filter.sum() > 0 and reset_timestep > 0:
            opacity = gaussians.get_opacity[visibility_filter]
            opa_loss = opt.opacity_loss_weight * torch.log(1 + opacity ** 2 / 0.5).mean()
            loss = loss + opa_loss

        if opt.lambda_curve_smo > 0 and visibility_filter.sum() > 0:
            rotation_mat = gaussians.get_rotation_matrix
            dir_global = rearrange(rotation_mat[..., 0], '(b m) c -> b m c', m=gaussians.n_gaussians)
            cos_sim = 1-F.cosine_similarity(dir_global[:, :-1, :], dir_global[:, 1:, :], dim=-1).abs()
            curve_smo = cos_sim.mean()
            loss = loss + opt.lambda_curve_smo * curve_smo

        if opt.lambda_width>0:
                width_thr = 0.005
                mask = gaussians.get_curve_width>=width_thr
                if mask.any():
                    lambda_width = (gaussians.get_curve_width[mask]-width_thr).mean()
                    loss = loss + opt.lambda_width * lambda_width

        if opt.lambda_points_conn > 0 and iteration > opt.conn_from_iter:
            curve_points = gaussians.get_curve_points
            start_points, end_points = curve_points[:,0], curve_points[:, -1] # N*3
            all_points = torch.cat([start_points, end_points], dim=0)
            mask = torch.eye(len(start_points), dtype=torch.bool, device=start_points.device)
            mask = torch.cat([torch.cat([mask, mask], dim=1), torch.cat([mask, mask], dim=1)], dim=0)
            dist = torch.cdist(all_points, all_points, p=2)
            dis_thr = 0.05
            with torch.no_grad():
                valid_mask = (dist < dis_thr) & (~mask)
            if valid_mask.any():
                curve_conn = dist[valid_mask] 
                curve_conn = curve_conn.mean()
                loss = loss + opt.lambda_points_conn * curve_conn
         
        loss.backward()
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_normal_for_log = 0.4 * normal_prior_error.item() + 0.6 * ema_normal_for_log
            ema_product_for_log = 0.4 * dot_products.item() + 0.6 * ema_product_for_log
            curve_smo_for_log = 0.4 * curve_smo.item() + 0.6 * curve_smo_for_log
            curve_conn_for_log = 0.4 * curve_conn.item() + 0.6 * curve_conn_for_log

            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "curve_smo": f"{curve_smo_for_log:.{5}f}",
                    "curve_conn": f"{curve_conn_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}",
                    "opacity": f"{gaussians.get_opacity.mean().item():.{5}f}",
                }
                progress_bar.set_postfix(loss_dict)
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            if tb_writer is not None:
                tb_writer.add_scalar('train_loss_patches/curve_smo', curve_smo_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/curve_conn', curve_conn_for_log, iteration)


            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
                            testing_iterations, scene, render,
                            (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp),
                            dataset.train_test_exp)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent, size_threshold, radii)



            if iteration==opt.densify_until_iter:
                prune_mask = (gaussians.get_curve_opacity <= opt.opacity_cull_second).squeeze()
                gaussians.prune_curves(prune_mask)
                torch.cuda.empty_cache()
                gaussians.fix_opacity()


            if iteration % 1000 == 500 and iteration > opt.densify_until_iter:
                gaussians.only_prune(opt.opacity_cull, opt.mask_threshold)
                gaussians.mask_trim_split(opt.mask_threshold)

            if iteration % 1000 ==0 and iteration > 3000 and iteration!=opt.iterations:
                gaussians.curve_split_curvature(opt.threshold_angle, opt.threshold_angle_skip)

            if (iteration % 1000 == 0 and iteration > opt.densify_until_iter) or iteration==opt.iterations:
                gaussians.fit_curve_to_line(opt.threshold_line, opt.threshold_max_line)
                gaussians.merge_curves(opt.distance_threshold, opt.similarity_threshold)

            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                gaussians.draw_curve(os.path.join(scene.model_path,
                                                           "point_cloud/iteration_{}".format(iteration)), iteration)

                gaussians.draw_ellipsoids(os.path.join(scene.model_path,
                                                           "point_cloud/iteration_{}".format(iteration)), iteration)

            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

        if hasattr(gaussians, 'prepare_scaling_rot'):
            gaussians.prepare_scaling_rot()
    



    extract_curves(gaussians, opt, scene)

def extract_curves(gaussians, opt, scene):
     # meger_points:
    merged_bezier_curves = gaussians.get_curve_points[gaussians.is_bezier]
    merged_line_segments = gaussians.get_curve_points[~gaussians.is_bezier][:,[0,-1],:]
    merged_bezier_curves = rearrange(merged_bezier_curves, 'b m c -> b (m c)').detach().cpu().numpy()
    merged_line_segments = rearrange(merged_line_segments, 'b m c -> b (m c)').detach().cpu().numpy()

    if opt.merge_endpoints_flag:
        (
            merged_line_segments,
            merged_bezier_curves,
        ) = merge_endpoints(
            merged_line_segments,
            merged_bezier_curves,
            distance_threshold=0.015,
        )

    merged_edge_dict = {
        "lines_end_pts": (
            merged_line_segments.tolist() if len(merged_line_segments) > 0 else []
        ),
        "curves_ctl_pts": (
            merged_bezier_curves.tolist() if len(merged_bezier_curves) > 0 else []
        ),
    }
    from edge_extraction.extract_para_edge import get_parametric_edge
    pred_edge_points, return_edge_dict = get_parametric_edge(opt.visible_checking, merged_edge_dict)
    edge_pcd = o3d.geometry.PointCloud()
    edge_pcd.points = o3d.utility.Vector3dVector(pred_edge_points)

    edge_ply_file_path = os.path.join(scene.model_path, "edge_points.ply")
    try:
        o3d.io.write_point_cloud(edge_ply_file_path, edge_pcd, write_ascii=True)
        print(f"Saved {edge_ply_file_path} for edge points visualization.")
    except IOError as e:
        print(f"Failed to save {edge_ply_file_path}: {e}")

    json_file_path = os.path.join(scene.model_path, "parametric_edges.json")
    try:
        with open(json_file_path, "w") as json_file:
            json.dump(return_edge_dict, json_file)
        print(f"Saved {json_file_path} for evaluation.")
    except IOError as e:
        print(f"Failed to save {json_file_path}: {e}")


def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        if args.detector=='DexiNed':
            args.model_path = os.path.join("./output_DexiNed/", unique_str[0:10])
        elif args.detector=='PidiNet':
            args.model_path = os.path.join("./output_PidiNet/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations,
                    scene : Scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                        from utils.general_utils import colormap
                        depth = render_pkg["depth"]
                        norm = depth.max()
                        depth = depth / norm
                        depth = colormap(depth.cpu().numpy()[0], cmap='turbo')
                        tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name),
                                             depth[None], global_step=iteration)

                        rend_alpha = render_pkg['rend_alpha']
                        rend_dir = F.normalize(render_pkg["rend_dir"], dim=0)
                        rend_dir = rend_dir * 0.5 + 0.5
                        tb_writer.add_images(config['name'] + "_view_{}/rend_dir".format(viewpoint.image_name),
                                             rend_dir[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/rend_alpha".format(viewpoint.image_name),
                                             rend_alpha[None], global_step=iteration)

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)


        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters", conflict_handler='resolve')
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6011)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[3_000, 10_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[3_000, 10_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[10000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    
    if 'ABC' in args.source_path:
        print("start training ABC")
        if lp.detector=='Pidinet':
            op = OptimizationParamsPidinet(parser)
    if 'Replica' in args.source_path:
        print("start training Replica")
        op = OptimizationParamsReplica(parser)

    args.save_iterations.append(args.iterations)
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), 
            args.test_iterations, args.save_iterations,
            args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
