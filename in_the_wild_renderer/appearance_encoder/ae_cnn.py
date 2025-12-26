import numpy as np
import torch
import cv2
import torch.nn as nn
from in_the_wild_renderer.appearance_encoder.ae_base import AEBase
from abc import abstractmethod
from typing import Optional
from jaxtyping import Shaped
from torch import Tensor, nn
from utils.appencoder_utils import PositionalEncoding, Embedding

class AECNN(AEBase):
    def __init__(self,opt,scene):
        super().__init__(opt,scene)
        num_train_cameras= len(scene.getTrainCameras())
        num_test_cameras= len(scene.getTestCameras())
        num_cameras = num_test_cameras + num_train_cameras
        input_channels = 4 if (opt.encode_with_mask==True and opt.mask_func=='cat') else 3
        self.mask_func = opt.mask_func
        self.encode_with_mask = opt.encode_with_mask
        self.encoder = Encoder(input_channels,opt.appearance_dim)
        self.trans_feat_dim = 3*(scene.gaussians.max_sh_degree + 1)**2
        self.pe_freq_feat = opt.app_pe_freq
        self.pe_freq_xyz = opt.pe_freq_xyz
        self.transnet = TransNet(self.trans_feat_dim,self.pe_freq_feat,pe_freq_xyz=self.pe_freq_xyz,appearance_dim=opt.appearance_dim)
    def forward(self,viewpoint_camera,pc,self_embeddings = None,mask = None, mask_feat=None):
        if self_embeddings !=None:
            embeddings = self_embeddings
        else:
            embeddings = self.encode(viewpoint_camera,mask,mask_feat)

        transed_dc = self.transfer(embeddings,pc)
        return transed_dc
    def encode(self,viewpoint_camera,mask=None, mask_feat=None):
        cam_raw_feat = None
        if hasattr(viewpoint_camera,'resized_image'):
            cam_raw_feat = viewpoint_camera.resized_image
        else:
            from torchvision import transforms
            transform = transforms.Resize([256, 256])
            cam_raw_feat = transform(viewpoint_camera.unsqueeze(0)).squeeze(0)
        # cam_raw_feat = cam_raw_feat.to('cuda') # ADD
        if self.encode_with_mask==1 and self.mask_func=='cat':
            from torchvision import transforms
            transform = transforms.Resize([256, 256])
            mask =transform(mask.unsqueeze(0)).squeeze(0)

            cam_raw_feat = torch.cat([cam_raw_feat,mask],dim=0)
        elif self.encode_with_mask==1 and self.mask_func=='multiply':
            from torchvision import transforms
            transform = transforms.Resize([256, 256])
            mask = transform(mask.unsqueeze(0)).squeeze(0)

            cam_raw_feat = cam_raw_feat * mask

        embedding = self.encoder(cam_raw_feat)
        return embedding
    def transfer(self,embeddings,pc):
        N = (pc.get_xyz).shape[0]
        # [ptr,SHs+Embeddings]->MLP->[ptr,SHs]

        feat = torch.cat([pc._features_dc, pc._features_rest], dim=-2)
        feat = feat.view(N,-1)
        feat = torch.cat([feat,pc.get_xyz],dim=-1) #!!!
        x_residual = self.transnet(feat,embeddings)

        return  x_residual
    def training_setup(self,opt):
        l = [
            {'params':self.transnet.parameters(),'lr':opt.transnet_lr,"name":"ae_transnet"},
            {'params': self.encoder.parameters(), 'lr': opt.encodenet_lr, "name": "ae_encoder"},
        ]
        self.optimizer = torch.optim.Adam(l,lr=1e-4,eps=1e-15)

    def save_checkpoint(self,path):
        torch.save(self.state_dict(),path)
    def load_latest_checkpoint(self,file_path):
        import os
        pth_files = [f for f in os.listdir(file_path) if f.endswith('.pth') and f.startswith('chkpnt_ae')]
        max_file_name = max(pth_files, key=lambda x: int(x.split('_')[-1][:-4]))
        self.load_state_dict(torch.load(os.path.join(file_path,max_file_name)))

class Encoder(nn.Module):
        def __init__(self, input_dim_a, output_nc=8,mid_dim=128):
            super().__init__()
            dim = mid_dim
            self.model = nn.Sequential(
                nn.ReflectionPad2d(3),
                nn.Conv2d(input_dim_a, dim, 7, 1),
                nn.ReLU(inplace=True),  ## size
                nn.ReflectionPad2d(1),
                nn.Conv2d(dim, dim * 2, 4, 2),
                nn.ReLU(inplace=True),  ## size/2
                nn.ReflectionPad2d(1),
                nn.Conv2d(dim * 2, dim * 4, 4, 2),
                nn.ReLU(inplace=True),  ## size/4
                nn.ReflectionPad2d(1),
                nn.Conv2d(dim * 4, dim * 4, 4, 2),
                nn.ReLU(inplace=True),  ## size/8
                nn.ReflectionPad2d(1),
                nn.Conv2d(dim * 4, dim * 4, 4, 2),
                nn.ReLU(inplace=True),  ## size/16
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(dim * 4, output_nc, 1, 1, 0)).cuda()  ## 1*1
            return

        def forward(self, x):
            x = self.model(x)
            output = x.view(x.size(0))
            return output

class TransNet(nn.Module):
    def __init__(self,trans_feat_dim,pe_freq_feat,pe_freq_xyz=4,appearance_dim=16,is_xyz=True):
        super().__init__()
        self.pe_freq_feat = pe_freq_feat
        self.pe_freq_xyz = pe_freq_xyz
        self.trans_feat_dim = trans_feat_dim
        self.appearance_dim = appearance_dim
        self.is_xyz =is_xyz
        self.pe_feat = PositionalEncoding(self.pe_freq_feat -1  ,self.pe_freq_feat)
        self.pe_xyz = PositionalEncoding(self.pe_freq_xyz - 1,self.pe_freq_xyz)
        self.mlp1 = nn.Linear((self.trans_feat_dim)*(self.pe_freq_feat*2+1)+int(is_xyz)*3*(self.pe_freq_xyz*2+1)+self.appearance_dim,256).cuda()# ！！ here add xyz
        self.mlp2 = nn.Linear(256,256).cuda()
        self.mlp3 = nn.Linear(256,256).cuda()
        self.mlp4 = nn.Linear((self.trans_feat_dim)*(self.pe_freq_feat*2+1)+int(is_xyz)*3*(self.pe_freq_xyz*2+1)+self.appearance_dim+256, 256).cuda()
        self.mlp5 = nn.Linear(256, 256).cuda()
        self.mlp6 = nn.Linear(256, 256).cuda()
        self.mlp7 = nn.Linear(256+self.trans_feat_dim, 128).cuda()
        self.mlp8 = nn.Linear(128, self.trans_feat_dim).cuda()
        self.batch_size = 65536*3
    def forward(self,x,embeddings):

        x_feat = x[...,:-3]
        x_feat_pe = self.pe_feat(x[...,:-3])  # [N,(1+15)*3]
        x_xyz = self.pe_xyz(x[...,-3:])  # [N,3*4]
        if self.is_xyz:
            x = torch.cat([x_feat_pe,embeddings.repeat([x.shape[0], 1]),x_xyz],dim=-1)
        else:
            x = torch.cat([x_feat_pe,embeddings.repeat([x.shape[0], 1])],dim=-1)

        outputs = []
        total_batches = (x.shape[0] + self.batch_size - 1) // self.batch_size
        for i in range(total_batches):
            x_feat_batch = x_feat[i*self.batch_size:(i+1)*self.batch_size]
            x_batch = x[i*self.batch_size:(i+1)*self.batch_size]
            x_batch_in = x_batch
            x_batch = torch.relu(self.mlp1(x_batch))
            x_batch = torch.relu(self.mlp2(x_batch))
            x_batch = torch.relu(self.mlp3(x_batch))
            x_batch = torch.relu(self.mlp4(torch.cat([x_batch_in,x_batch],dim=-1)))
            x_batch = torch.relu(self.mlp5(x_batch))
            x_batch = torch.relu(self.mlp6(x_batch))
            x_batch = torch.relu(self.mlp7(torch.cat([x_feat_batch,x_batch],dim=-1)))
            x_batch = torch.tanh(self.mlp8(x_batch))
            # x_batch = (self.mlp8(x_batch))
            outputs.append(x_batch)
        outputs = torch.cat(outputs,dim=0)
        outputs = outputs.view(outputs.shape[0],-1,3)
        return outputs[:,...]

