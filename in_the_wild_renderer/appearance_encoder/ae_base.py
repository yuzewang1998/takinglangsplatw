import numpy as np
import torch
import cv2
import torch.nn as nn

class AEBase(nn.Module):
    def __init__(self,opt,scene):
        super().__init__()
        self.appearance_dim = opt.appearance_dim
    def forward(self,viewpoint_camera):
        pass