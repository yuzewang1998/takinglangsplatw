import numpy as np
import torch
import cv2
import torch.nn as nn

class TEBase(nn.Module):
    def __init__(self,opt,scene):
        super().__init__()
        self.transient_dim = opt.transient_dim
    def forward(self,viewpoint_camera):
        pass