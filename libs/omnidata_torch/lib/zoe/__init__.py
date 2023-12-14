# ZoeDepth
# https://github.com/isl-org/ZoeDepth

import os
import cv2
import numpy as np
import torch
import math
import os
import requests
from torch.hub import download_url_to_file, get_dir
from tqdm import tqdm
from urllib.parse import urlparse
import torch.nn as nn
import pdb

from einops import rearrange
from .zoedepth.models.zoedepth.zoedepth_v1 import ZoeDepth
from .zoedepth.utils.config import get_config
annotator_ckpts_path = '/mnt/workspace/codes/omnidata/omnidata_tools/torch/pretrained_models'

def load_file_from_url(url, model_dir=None, progress=True, file_name=None):
    """Load file form http url, will download models if necessary.

    Ref:https://github.com/1adrianb/face-alignment/blob/master/face_alignment/utils.py

    Args:
        url (str): URL to be downloaded.
        model_dir (str): The path to save the downloaded model. Should be a full path. If None, use pytorch hub_dir.
            Default: None.
        progress (bool): Whether to show the download progress. Default: True.
        file_name (str): The downloaded file name. If None, use the file name in the url. Default: None.

    Returns:
        str: The path to the downloaded file.
    """
    if model_dir is None:  # use the pytorch hub_dir
        hub_dir = get_dir()
        model_dir = os.path.join(hub_dir, 'checkpoints')

    os.makedirs(model_dir, exist_ok=True)

    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    if file_name is not None:
        filename = file_name
    cached_file = os.path.abspath(os.path.join(model_dir, filename))
    if not os.path.exists(cached_file):
        print(f'Downloading: "{url}" to {cached_file}\n')
        download_url_to_file(url, cached_file, hash_prefix=None, progress=progress)
    return cached_file




class ZoeDetector:
    def __init__(self):
        remote_model_path = "https://huggingface.co/lllyasviel/Annotators/resolve/main/ZoeD_M12_N.pt"
        modelpath = os.path.join(annotator_ckpts_path, "ZoeD_M12_N.pt")

        if not os.path.exists(modelpath):
            load_file_from_url(remote_model_path, model_dir=annotator_ckpts_path)
        conf = get_config("zoedepth", "infer")
        model = ZoeDepth.build_from_config(conf)
        model.load_state_dict(torch.load(modelpath)['model'])
        model = model.cuda()
        model.device = 'cuda'
        model.eval()
        self.model = model

    def __call__(self, input_image):
        assert input_image.ndim == 3
        image_depth = input_image
        with torch.no_grad():
            image_depth = torch.from_numpy(image_depth).float().cuda()
            image_depth = image_depth / 255.0
            image_depth = rearrange(image_depth, 'h w c -> 1 c h w')
            depth = self.model.infer(image_depth)
            __depth = depth.clone()

            depth = depth[0, 0].cpu().numpy()

            vmin = np.percentile(depth, 2)
            vmax = np.percentile(depth, 85)

            depth -= vmin
            depth /= vmax - vmin
            depth = 1.0 - depth
            depth_image = (depth * 255.0).clip(0, 255).astype(np.uint8)

            return depth_image, __depth

class ZoeBatchDetector(nn.Module):
    def __init__(self):
        super().__init__()

        self.z_near = 1.
        remote_model_path = "https://huggingface.co/lllyasviel/Annotators/resolve/main/ZoeD_M12_N.pt"
        modelpath = os.path.join(annotator_ckpts_path, "ZoeD_M12_N.pt")

        if not os.path.exists(modelpath):
            load_file_from_url(remote_model_path, model_dir=annotator_ckpts_path)
        conf = get_config("zoedepth", "infer")
        model = ZoeDepth.build_from_config(conf)
        model.load_state_dict(torch.load(modelpath)['model'])
        model.eval()
        self.model = model

    def forward(self, input_image):
        assert input_image.ndim == 4
        image_depth = input_image

        with torch.no_grad():
            image_depth = image_depth / 255.0
            batch = image_depth.shape[0]
            image_depth = rearrange(image_depth, 'b h w c -> b c h w')
            depth = self.model.infer(image_depth).clamp(min=0.)[:, 0, ...]

            depth += self.z_near
            disparity = 1./depth

            min_v = torch.min(disparity.view(batch,-1),dim = 1, keepdim=True)[0]
            max_v = torch.max(disparity.view(batch,-1),dim = 1, keepdim=True)[0]

            disparity -= min_v[...,None]
            disparity /= (max_v[...,None] - min_v[..., None])

            disparity = disparity * 2 - 1

            return disparity[:, None, ...]
