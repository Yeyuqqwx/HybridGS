from . import gsplat
from .gsplat.gsplat.project_gaussians_2d import project_gaussians_2d
from .gsplat.gsplat.rasterize_sum import rasterize_gaussians_sum
import torch
import torch.nn as nn

class GaussianImage(nn.Module):
    def __init__(self, init_num_points=10000, H=391, W=530, active_uncertainty=True, device='cuda'):
        super().__init__()
        self.init_num_points = init_num_points
        self.H, self.W = H, W
        self.BLOCK_W, self.BLOCK_H = 16, 16
        self.tile_bounds = (
            (self.W + self.BLOCK_W - 1) // self.BLOCK_W,
            (self.H + self.BLOCK_H - 1) // self.BLOCK_H,
            1,
        ) # 
        self.device = device
        self._xyz = torch.atanh(2 * (torch.rand(self.init_num_points, 2) - 0.5)) # n,2
        self._cholesky = torch.rand(self.init_num_points, 3) # n,3
        self._features_dc = torch.rand(self.init_num_points, 3) # n,3
        self.active_uncertainty = active_uncertainty
        if self.active_uncertainty:
            self._uncertainty = torch.logit(0.5 * torch.ones(self.init_num_points, 1))
        else:
            self.register_buffer('_uncertainty', torch.ones((self.init_num_points, 1)))
        self.register_buffer('_opacity', torch.ones((self.init_num_points, 1)))
        self.register_buffer('background', torch.ones(3))
        self.register_buffer('bound', torch.tensor([0.5, 0.5]).view(1, 2))
        self.register_buffer('cholesky_bound', torch.tensor([0.5, 0, 0.5]).view(1, 3))
    @property
    def get_xyz(self):
        return torch.tanh(self._xyz)
    @property
    def get_features(self):
        return self._features_dc
    @property
    def get_uncertainty(self):
        if self.active_uncertainty:
            return torch.sigmoid(self._uncertainty)
        else:
            return self._uncertainty
    @property
    def get_cholesky_elements(self):
        return self._cholesky+self.cholesky_bound
    def forward(self, embed, img_size=(512, 512), return_mask=False):
        embed = embed.view(-1, 9)
        self._xyz = embed[:, :2] # n,2
        self._features_dc = embed[:, 2:5] # n,3
        self._cholesky = embed[:, 5:8] # n,3
        self._uncertainty = embed[:, 8:] # n,1
        self.H, self.W = img_size
        self.tile_bounds = (
            (self.W + self.BLOCK_W - 1) // self.BLOCK_W,
            (self.H + self.BLOCK_H - 1) // self.BLOCK_H,
            1,
        ) 
        
        self.xys, depths, self.radii, conics, num_tiles_hit = project_gaussians_2d(self.get_xyz, self.get_cholesky_elements, self.H, self.W, self.tile_bounds)
        out_img = rasterize_gaussians_sum(self.xys, depths, self.radii, conics, num_tiles_hit,
            self.get_features * self.get_uncertainty, self._opacity, self.H, self.W, self.BLOCK_H, self.BLOCK_W, background=self.background)
        out_img = out_img.view(-1, self.H, self.W, 3).permute(0, 3, 1, 2).contiguous()
        if return_mask:
            out_alpha = rasterize_gaussians_sum(self.xys, depths, self.radii, conics, num_tiles_hit,
                torch.ones(self.init_num_points, 3, device=self.device) * self.get_uncertainty, self._opacity, self.H, self.W, self.BLOCK_H, self.BLOCK_W, background=self.background)
            out_alpha = out_alpha.view(-1, self.H, self.W, 3).permute(0, 3, 1, 2).contiguous()
            return out_img, out_alpha
        
        return out_img