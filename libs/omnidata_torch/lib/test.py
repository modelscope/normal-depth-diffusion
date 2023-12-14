# -*- coding: utf-8 -*-

import pdb
from midas_31 import MidasBatchDetector, MidasDetector
import cv2
from PIL import Image
import numpy as np
import time
import torch
from einops import repeat

model = MidasBatchDetector().cuda()

img = cv2.imread('./lion.png')[...,[2,1,0]]
img = cv2.resize(img,(384, 384))
img = torch.from_numpy(img).float()

img = repeat(img, 'h w c-> b h w c', b=8).cuda()

start = time.time()
with torch.no_grad():
    ret_vals = model(img)
depth= ret_vals['depth']
depth = (depth+1)/2 * 255
depth=depth.permute(0,2,3,1).squeeze(-1)
depth = depth.detach().cpu().numpy().astype(np.uint8)

print("cost:{:.4f} s".format(time.time()-start))


cv2.imwrite("depth.png", depth[0])
pdb.set_trace()






































