import json
import math
import os
import time
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union
from gsplat.strategy import DefaultStrategy, MCMCStrategy
import imageio
import numpy as np
from tqdm import tqdm
import torch
import tyro
from datasets.colmap import Parser, ClutterDataset
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from fused_ssim import fused_ssim
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import Literal, assert_never
from utils.utils import AppearanceOptModule, CameraOptModule, knn, rgb_to_sh, set_random_seed

from gsplat.distributed import cli
from gsplat.rendering import rasterization
from gsplat.optimizers import SelectiveAdam

from gimage.gaussianimage import GaussianImage

@dataclass
class Config:
    metric_only: bool = False
    gimage: bool = False
    # Path to the .pt files. If provide, it will skip training and run evaluation only.
    ckpt_path: Optional[str] = None


    # Path to the Mip-NeRF 360 dataset
    data_dir: str = "data/360_v2/garden"
    # Downsample factor for the dataset
    data_factor: int = 4
    # Directory to save results
    result_dir: str = "results/garden"
    # Every N images there is a test image
    test_every: int = 8
    # Random crop size for training  (experimental)
    patch_size: Optional[int] = None
    # A global scaler that applies to the scene size related parameters
    global_scale: float = 1.0
    # Normalize the world space
    normalize_world_space: bool = True
    # Camera model
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole"

    # Port for the viewer server
    port: int = 8080

    # Batch size for training. Learning rates are scaled automatically
    batch_size: int = 1
    # A global factor to scale the number of training steps
    steps_scaler: float = 1.0

    # Number of training steps
    max_steps: int = 30_000

    eval_steps: List[int] = field(default_factory=lambda: [20, 200, 400, 800, 1200, 2000])
    # Steps to save the model
    save_steps: List[int] = field(default_factory=lambda: [200, 400, 800, 1200, 2000])


    # Initialization strategy
    init_type: str = "sfm"
    # Initial number of GSs. Ignored if using sfm
    init_num_pts: int = 100_000
    # Initial extent of GSs as a multiple of the camera extent. Ignored if using sfm
    init_extent: float = 3.0
    # Degree of spherical harmonics
    sh_degree: int = 3
    # Turn on another SH degree every this steps
    sh_degree_interval: int = 1000
    # Initial opacity of GS
    init_opa: float = 0.1
    # Initial scale of GS
    init_scale: float = 1.0
    # Weight for SSIM loss
    ssim_lambda: float = 0.2

    # Near plane clipping distance
    near_plane: float = 0.01
    # Far plane clipping distance
    far_plane: float = 1e10
     # Strategy for GS densification
    strategy: Union[DefaultStrategy, MCMCStrategy] = field(
        default_factory=DefaultStrategy
    )
    # Use packed mode for rasterization, this leads to less memory usage but slightly slower.
    packed: bool = False
    # Use sparse gradients for optimization. (experimental)
    sparse_grad: bool = False
    # Use visible adam from Taming 3DGS. (experimental)
    visible_adam: bool = False
    # Anti-aliasing in rasterization. Might slightly hurt quantitative metrics.
    antialiased: bool = False

    # Use random background for training to discourage transparency
    random_bkgd: bool = False

    # Opacity regularization
    opacity_reg: float = 0.0
    # Scale regularization
    scale_reg: float = 0.0

    # Enable camera optimization.
    pose_opt: bool = False
    # Learning rate for camera optimization
    pose_opt_lr: float = 1e-5
    # Regularization for camera optimization as weight decay
    pose_opt_reg: float = 1e-6
    # Add noise to camera extrinsics. This is only to test the camera pose optimization.
    pose_noise: float = 0.0

    # Enable appearance optimization. (experimental)
    app_opt: bool = False
    # Appearance embedding dimension
    app_embed_dim: int = 16
    # Learning rate for appearance optimization
    app_opt_lr: float = 1e-3
    # Regularization for appearance optimization as weight decay
    app_opt_reg: float = 1e-6
    # Shape of the bilateral grid (X, Y, W)
    bilateral_grid_shape: Tuple[int, int, int] = (16, 16, 8)

    # Enable depth loss. (experimental)
    depth_loss: bool = False
    # Weight for depth loss
    depth_lambda: float = 1e-2

    # Dump information to tensorboard every this steps
    tb_every: int = 100
    # Save training images to tensorboard
    tb_save_image: bool = False

    lpips_net: Literal["vgg", "alex"] = "alex"


def create_splats_with_optimizers(
    parser: Parser,
    init_type: str = "sfm",
    init_num_pts: int = 100_000,
    init_extent: float = 3.0,
    init_opacity: float = 0.1,
    init_scale: float = 1.0,
    scene_scale: float = 1.0,
    sh_degree: int = 3,
    sparse_grad: bool = False,
    visible_adam: bool = False,
    batch_size: int = 1,
    feature_dim: Optional[int] = None,
    device: str = "cuda",
    world_rank: int = 0,
    world_size: int = 1,
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:
    if init_type == "sfm":
        points = torch.from_numpy(parser.points).float()
        rgbs = torch.from_numpy(parser.points_rgb / 255.0).float()
    elif init_type == "random":
        points = init_extent * scene_scale * (torch.rand((init_num_pts, 3)) * 2 - 1)
        rgbs = torch.rand((init_num_pts, 3))
    else:
        raise ValueError("Please specify a correct init_type: sfm or random")

    # Initialize the GS size to be the average dist of the 3 nearest neighbors
    dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
    dist_avg = torch.sqrt(dist2_avg)
    scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)  # [N, 3]

    # Distribute the GSs to different ranks (also works for single rank)
    points = points[world_rank::world_size]
    rgbs = rgbs[world_rank::world_size]
    scales = scales[world_rank::world_size]

    N = points.shape[0]
    quats = torch.rand((N, 4))  # [N, 4]
    opacities = torch.logit(torch.full((N,), init_opacity))  # [N,]

    params = [
        # name, value, lr
        ("means", torch.nn.Parameter(points), 1.6e-4 * scene_scale),
        ("scales", torch.nn.Parameter(scales), 5e-3),
        ("quats", torch.nn.Parameter(quats), 1e-3),
        ("opacities", torch.nn.Parameter(opacities), 5e-2),
    ]

    if feature_dim is None:
        # color is SH coefficients.
        colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))  # [N, K, 3]
        colors[:, 0, :] = rgb_to_sh(rgbs)
        params.append(("sh0", torch.nn.Parameter(colors[:, :1, :]), 2.5e-3))
        params.append(("shN", torch.nn.Parameter(colors[:, 1:, :]), 2.5e-3 / 20))
    else:
        # features will be used for appearance and view-dependent shading
        features = torch.rand(N, feature_dim)  # [N, feature_dim]
        params.append(("features", torch.nn.Parameter(features), 2.5e-3))
        colors = torch.logit(rgbs)  # [N, 3]
        params.append(("colors", torch.nn.Parameter(colors), 2.5e-3))

    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)
    # Scale learning rate based on batch size, reference:
    # https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
    # Note that this would not make the training exactly equivalent, see
    # https://arxiv.org/pdf/2402.18824v1
    BS = batch_size * world_size
    optimizer_class = None
    if sparse_grad:
        optimizer_class = torch.optim.SparseAdam
    elif visible_adam:
        optimizer_class = SelectiveAdam
    else:
        optimizer_class = torch.optim.Adam
    optimizers = {
        name: optimizer_class(
            [{"params": splats[name], "lr": lr * math.sqrt(BS), "name": name}],
            eps=1e-15 / math.sqrt(BS),
            # TODO: check betas logic when BS is larger than 10 betas[0] will be zero.
            betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
        )
        for name, _, lr in params
    }
    return splats, optimizers


class Runner:
    """Engine for training and testing."""

    def __init__(
        self, local_rank: int, world_rank, world_size: int, cfg: Config
    ) -> None:
        set_random_seed(42 + local_rank)

        self.cfg = cfg
        self.world_rank = world_rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = f"cuda:{local_rank}"
        self.scene_name = cfg.data_dir.split("/")[-1]


        if cfg.ckpt_path is not None:
            cfg.result_dir = cfg.ckpt_path
        # Where to dump results.
       
        # import pdb; pdb.set_trace()
        os.makedirs(cfg.result_dir, exist_ok=True)

        # Setup output directories.
        self.stats_dir = f"{cfg.result_dir}"
        os.makedirs(self.stats_dir, exist_ok=True)
        self.render_dir = f"{cfg.result_dir}/renders"
        os.makedirs(self.render_dir, exist_ok=True)

        # Load data: Training data should contain initial points and colors.
        self.parser = Parser(
            data_dir=cfg.data_dir,
            factor=cfg.data_factor,
            normalize=cfg.normalize_world_space,
            test_every=cfg.test_every,
        )
        self.trainset = ClutterDataset(
            self.parser,
            split="train",
            patch_size=cfg.patch_size,
            load_depths=cfg.depth_loss,
        )
        self.valset = ClutterDataset(self.parser, split="val")


        self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale
        print("Scene scale:", self.scene_scale)

        # Model
        feature_dim = 32 if cfg.app_opt else None
        self.splats, self.optimizers = create_splats_with_optimizers(
            self.parser,
            init_type=cfg.init_type,
            init_num_pts=cfg.init_num_pts,
            init_extent=cfg.init_extent,
            init_opacity=cfg.init_opa,
            init_scale=cfg.init_scale,
            scene_scale=self.scene_scale,
            sh_degree=cfg.sh_degree,
            sparse_grad=cfg.sparse_grad,
            visible_adam=cfg.visible_adam,
            batch_size=cfg.batch_size,
            feature_dim=feature_dim,
            device=self.device,
            world_rank=world_rank,
            world_size=world_size,
        )
        print("Model initialized. Number of GS:", len(self.splats["means"]))

        self.app_optimizers = []
        if cfg.app_opt:
            assert feature_dim is not None
            self.app_module = AppearanceOptModule(
                len(self.trainset), feature_dim, cfg.app_embed_dim, cfg.sh_degree
            ).to(self.device)
            # initialize the last layer to be zero so that the initial output is zero.
            torch.nn.init.zeros_(self.app_module.color_head[-1].weight)
            torch.nn.init.zeros_(self.app_module.color_head[-1].bias)
            self.app_optimizers = [
                torch.optim.Adam(
                    self.app_module.embeds.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size) * 10.0,
                    weight_decay=cfg.app_opt_reg,
                ),
                torch.optim.Adam(
                    self.app_module.color_head.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size),
                ),
            ]
            if world_size > 1:
                self.app_module = DDP(self.app_module)


        # gimage_embedding
        self.dim_embed = 10000
        dim_feat = 9
        num_cameras = len(self.trainset)
        self.embedding = torch.nn.Embedding(num_cameras, self.dim_embed * dim_feat).cuda()
        self.embedding.weight.data.view(num_cameras, self.dim_embed, dim_feat)[:, :, :2] = torch.atanh(
            torch.rand((num_cameras, self.dim_embed, 2)) * 2 - 1)  # xy
        self.embedding.weight.data.view(num_cameras, self.dim_embed, dim_feat)[:, :, 2:5] = torch.ones(
            (num_cameras, self.dim_embed, 3)) * 0.5  # rgb
        self.embedding.weight.data.view(num_cameras, self.dim_embed, dim_feat)[:, :, 5:8] = torch.rand(
            (num_cameras, self.dim_embed, 3))  # cholesky
        self.embedding.weight.data.view(num_cameras, self.dim_embed, dim_feat)[:, :, 8] = torch.logit(
            0.5 * torch.ones(num_cameras, self.dim_embed))  # opacity
        self.gaussianimage = GaussianImage(init_num_points=self.dim_embed, active_uncertainty=True).cuda()
        # Losses & Metrics.
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)

        if cfg.lpips_net == "alex":
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="alex", normalize=True
            ).to(self.device)
        elif cfg.lpips_net == "vgg":
            # The 3DGS official repo uses lpips vgg, which is equivalent with the following:
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="vgg", normalize=False
            ).to(self.device)
        else:
            raise ValueError(f"Unknown LPIPS network: {cfg.lpips_net}")


    def rasterize_splats(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        masks: Optional[Tensor] = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        means = self.splats["means"]  # [N, 3]
        # quats = F.normalize(self.splats["quats"], dim=-1)  # [N, 4]
        # rasterization does normalization internally
        quats = self.splats["quats"]  # [N, 4]
        scales = torch.exp(self.splats["scales"])  # [N, 3]
        opacities = torch.sigmoid(self.splats["opacities"])  # [N,]

        image_ids = kwargs.pop("image_ids", None)
        if self.cfg.app_opt:
            colors = self.app_module(
                features=self.splats["features"],
                embed_ids=image_ids,
                dirs=means[None, :, :] - camtoworlds[:, None, :3, 3],
                sh_degree=kwargs.pop("sh_degree", self.cfg.sh_degree),
            )
            colors = colors + self.splats["colors"]
            colors = torch.sigmoid(colors)
        else:
            colors = torch.cat([self.splats["sh0"], self.splats["shN"]], 1)  # [N, K, 3]

        rasterize_mode = "antialiased" if self.cfg.antialiased else "classic"
        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            packed=self.cfg.packed,
            absgrad=(
                self.cfg.strategy.absgrad
                if isinstance(self.cfg.strategy, DefaultStrategy)
                else False
            ),
            sparse_grad=self.cfg.sparse_grad,
            rasterize_mode=rasterize_mode,
            distributed=self.world_size > 1,
            camera_model=self.cfg.camera_model,
            **kwargs,
        )
        if masks is not None:
            render_colors[~masks] = 0
        return render_colors, render_alphas, info
    
    @torch.no_grad()
    def eval(self, stage: str = "test"):
        """Entry for evaluation."""
        print(f"Running evaluation {stage}...")
        cfg = self.cfg
        device = self.device
        stage_images_path = os.path.join(self.render_dir, stage)
        os.makedirs(stage_images_path, exist_ok=True)
        
        stage_r_s = os.path.join(stage_images_path, 'r_s')
        stage_gt = os.path.join(stage_images_path, 'gt')
        stage_cat = os.path.join(stage_images_path, 'cat')
        os.makedirs(stage_r_s, exist_ok=True)
        os.makedirs(stage_gt, exist_ok=True)
        os.makedirs(stage_cat, exist_ok=True)

        if stage == "train":
            stage_r_t = os.path.join(stage_images_path, 'r_t')
            stage_mask_t = os.path.join(stage_images_path, 'mask_t')    
            stage_r_st = os.path.join(stage_images_path, 'r_st')
            os.makedirs(stage_mask_t, exist_ok=True)
            os.makedirs(stage_r_t, exist_ok=True)
            os.makedirs(stage_r_st, exist_ok=True)
            loader = torch.utils.data.DataLoader(
                self.trainset, batch_size=1, shuffle=False, num_workers=1
            )
        else:
            loader = torch.utils.data.DataLoader(
                self.valset, batch_size=1, shuffle=False, num_workers=1
            )

        ellipse_time = 0
        metrics = defaultdict(list)
        for i, data in enumerate(tqdm(loader)):
            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            gt = data["image"].to(device) / 255.0
            masks = data["mask"].to(device) if "mask" in data else None
            height, width = gt.shape[1:3]
            torch.cuda.synchronize()
            tic = time.time()
            r_s, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                masks=masks,
            ) 
            torch.cuda.synchronize()
            ellipse_time += time.time() - tic

            r_s = torch.clamp(r_s, 0.0, 1.0)
            if stage == "train":
                image_ids = data["image_id"].to(device)
                embed = self.embedding(image_ids)
                r_t, mask_t = self.gaussianimage(embed, (r_s.shape[1], r_s.shape[2]), return_mask=True)
                r_t = torch.clamp(r_t, 0.0, 1.0)
                mask_t = torch.clamp(mask_t, 0.0, 1.0)
                r_t = r_t.permute(0, 2, 3, 1)
                mask_t = mask_t.permute(0, 2, 3, 1)
                r_st = r_t + (1 - mask_t) * r_s
                r_st = torch.clamp(r_st, 0.0, 1.0)
                cat = torch.cat([gt, r_st, r_s, r_t, mask_t], dim=1).squeeze(0).cpu().numpy()
                cat = (cat * 255).astype(np.uint8)
                imageio.imwrite(
                        f"{stage_r_t}/{stage}_{i:04d}.png",
                        (r_t.squeeze(0).cpu().numpy() * 255).astype(np.uint8),
                    )
                imageio.imwrite(
                        f"{stage_mask_t}/{stage}_{i:04d}.png",
                        (mask_t.squeeze(0).cpu().numpy() * 255).astype(np.uint8),
                    )
                imageio.imwrite(
                        f"{stage_r_st}/{stage}_{i:04d}.png",
                        (r_st.squeeze(0).cpu().numpy() * 255).astype(np.uint8),
                    )
            else:
                if not cfg.metric_only:
                    cat = torch.cat([gt, r_s], dim=1).squeeze(0).cpu().numpy()
                    cat = (cat * 255).astype(np.uint8)
            if not cfg.metric_only:
                imageio.imwrite(
                        f"{stage_cat}/{stage}_{i:04d}.png",
                        cat,
                    )
                imageio.imwrite(
                        f"{stage_gt}/{stage}_{i:04d}.png",
                        (gt.squeeze(0).cpu().numpy() * 255).astype(np.uint8),
                    )
                imageio.imwrite(
                        f"{stage_r_s}/{stage}_{i:04d}.png",
                        (r_s.squeeze(0).cpu().numpy() * 255).astype(np.uint8),
                    )
            
            gt_p = gt.permute(0, 3, 1, 2)  # [1, 3, H, W]
            if stage == "train":
                r_s_p = r_st.permute(0, 3, 1, 2)  # [1, 3, H, W]
            else:
                r_s_p = r_s.permute(0, 3, 1, 2)  # [1, 3, H, W]
            metrics["psnr"].append(self.psnr(r_s_p, gt_p))
            metrics["ssim"].append(self.ssim(r_s_p, gt_p))
            metrics["lpips"].append(self.lpips(r_s_p, gt_p))

        ellipse_time /= len(loader)

        stats = {k: torch.stack(v).mean().item() for k, v in metrics.items()}
        stats.update(
            {
                "ellipse_time": ellipse_time,
                "num_GS": len(self.splats["means"]),
            }
        )
        print(
            f"{stage}: "
            f"\033[94m PSNR: {stats['psnr']:.3f}, SSIM: {stats['ssim']:.4f}, LPIPS: {stats['lpips']:.3f}\033[0m "
            f"Time: {stats['ellipse_time']:.3f}s/image "
            f"Number of GS: {stats['num_GS']}"
        )
        # save stats as json
        with open(f"{self.stats_dir}/{stage}.json", "w") as f:
            json.dump(stats, f)


def main(local_rank: int, world_rank, world_size: int, cfg: Config):
    runner = Runner(local_rank, world_rank, world_size, cfg)

    if cfg.ckpt_path is not None:
        # run eval only
        ckpt = torch.load(os.path.join(cfg.ckpt_path, "splats.pt"), map_location=runner.device, weights_only=True)
        for k in runner.splats.keys():
            runner.splats[k] = ckpt[k]
        # runner.splats.data = ckpt
        runner.embedding = torch.jit.load(os.path.join(cfg.ckpt_path, 'embedding.pt')).cuda()
        runner.eval(stage="test")
        if not cfg.metric_only:
            runner.eval(stage="train")

if __name__ == "__main__":
    cfg = tyro.cli(Config)
    cli(main, cfg, verbose=True)