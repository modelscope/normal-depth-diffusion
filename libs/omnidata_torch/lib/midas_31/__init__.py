# Midas Depth Estimation
# From https://github.com/isl-org/MiDaS
# MIT LICENSE

import cv2
import numpy as np
import torch

from einops import rearrange
from .api import MiDaSInference
import pdb
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat



class MidasBatchDetectorW(object):
    def __init__(self):
        self.depth_prior = MidasBatchDetector()

    def train(self):
        self.depth_prior.train()
    def eval(self):
        self.depth_prior.eval()
    def to(self,device):
        self.depth_prior.to(device)
    def cuda(self):
        self.depth_prior.cuda()
    def cpu(self):
        self.depth_prior.cpu()

    def __call__(self,x):
        with torch.no_grad():
            return self.depth_prior(x)





def freeze(model):
    for para in model.parameters():
        para.requires_grad =False

class MidasDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = MiDaSInference(model_type="dpt_beit_large_512")

    def forward(self, input_image, a=np.pi*2.0, bg_th=0.1):
        assert input_image.ndim == 3
        image_depth = input_image
        with torch.no_grad():
            image_depth = torch.from_numpy(image_depth).float()
            image_depth = image_depth / 127.5 - 1.0
            image_depth = rearrange(image_depth, 'h w c -> 1 c h w')
            depth = self.model(image_depth)[0]

            depth2img = depth.clone()

            depth2img -= torch.min(depth2img)
            depth2img /= torch.max(depth2img)
            depth2img = depth2img.cpu().numpy()
            depth_image = (depth2img * 255.0).clip(0, 255).astype(np.uint8)


            return depth_image, depth

class Sobel(nn.Module):
    def __init__(self):
        super().__init__()
        # self.conv_X = nn.Conv2d(in_channels=1, out_channels=1,kernel_size=3,stride=1, padding=1, bias=False )
        # self.conv_Y = nn.Conv2d(in_channels=1, out_channels=1,kernel_size=3,stride=1, padding=1, bias= False)

        x_kernel = torch.tensor([[-1,0,1], [-2, 0, 2], [-1, 0, 1]]).float()
        y_kernel = torch.tensor([[-1,-2,-1], [0, 0, 0], [1, 2, 1]]).float()
        self.x_kernel = torch.nn.Parameter(x_kernel[None, None])
        self.y_kernel = torch.nn.Parameter(y_kernel[None, None])

        self.conv_X = lambda x: F.conv2d(x, self.x_kernel, bias= None, stride=1, padding= 0)
        self.conv_Y = lambda y: F.conv2d(y, self.y_kernel, bias= None, stride=1, padding =0)

        # freeze conv
        freeze(self)


    def forward(self, x, back_threshold = 1.):
        '''x is disparity
        '''
        _,_, h , w = x.shape

        # copy from controlNet
        nx = -self.conv_X(x)
        ny = -self.conv_Y(x)
        nz = torch.ones_like(nx) * 2 * torch.pi
        _,_, conv_h,conv_w = nx.shape

        conv_x = F.interpolate(x,(conv_w,conv_h),mode='bilinear', align_corners= True )

        nx[conv_x<back_threshold]=0
        ny[conv_x<back_threshold]=0

        normal = torch.cat([nx, ny, nz], dim=1)
        # nz is not zero,hence we not need safe_normalized

        normal = F.interpolate(normal,(w,h),mode='bilinear', align_corners= True )
        normal /= (normal.norm(dim=1,keepdim=True))

        return normal












class MidasBatchDetector(nn.Module):
    def __init__(self, need_normal = False):
        super().__init__()
        self.model = MiDaSInference(model_type="dpt_beit_large_512")
        self.need_normal = need_normal
        self.sobel_depth2normal = Sobel()



    def to_normal(self, depth_pt ,depth_threshold = 0.4):
        '''
        transfer_depth_to_normal
        depth_pt : [b, h,w]
        '''
        # NOTE that no use, as no better than normalbae. plenty of noise

        normal =  self.sobel_depth2normal(depth_pt[:,None,:,:], back_threshold = depth_threshold)

        return normal



    def forward(self, input_image):
        ret_vals={}
        assert input_image.ndim == 4
        image_depth = input_image
        with torch.no_grad():
            image_depth = image_depth / 127.5 - 1.0
            image_depth = rearrange(image_depth, 'b h w c -> b c h w')
            depth = self.model(image_depth)

            batch = depth.shape[0]

            min_v = torch.min(depth.view(batch,-1),dim = 1, keepdim=True)[0]
            max_v = torch.max(depth.view(batch,-1),dim = 1, keepdim=True)[0]

            normalized_depth= (depth - min_v[..., None]) / (max_v[...,None] - min_v[...,None])
            normalized_depth = normalized_depth.clamp(0,1)

            # normalized to [-1 1]

            if self.need_normal:
                # NOTE that normal value is ~[-1,1]
                normalized_normal  = self.to_normal(depth)
                ret_vals.update(
                        {'normal': normalized_normal}
                        )

            normalized_depth = normalized_depth * 2 - 1

            ret_vals.update(
                    {'depth': normalized_depth[:,None,...]}
                    )


            return ret_vals
