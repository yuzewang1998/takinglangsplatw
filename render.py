#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import numpy as np
import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import math
def eulerAnglesToRotationMatrix(theta):
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]), math.cos(theta[0])]
                    ])
    R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                    [0, 1, 0],
                    [-math.sin(theta[1]), 0, math.cos(theta[1])]
                    ])

    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]), math.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R
def constructEvalCameras(scene):
    # print(scene.model_path)
    # scene_name = scene.model_path.split('/')[-3]
    # print(scene_name)
    # Attention! now use a hard-code
    scene_name = 'trevi_fountain'
    seq=2
    # seq=1
    # scene_name = 'pantheon'
    if scene_name == 'pantheon' and seq == 1:
        N_FRAMES = 300
        eval_cameras = []
        ref_camera = scene.getTrainCameras()[3]  # test8
        print(ref_camera.colmap_id)
        print(ref_camera.R)
        print(ref_camera.T)

        dx1 = np.linspace(-1, 0., N_FRAMES // 2)
        dx2 = np.linspace(0, 0, N_FRAMES - N_FRAMES // 2)
        dx = np.concatenate((dx1, dx2))

        dy1 = np.linspace(-1., -1, N_FRAMES // 2)
        dy2 = np.linspace(-1., 0, N_FRAMES - N_FRAMES // 2)
        dy = np.concatenate((dy1, dy2))

        dz1 = np.linspace(1, 2, N_FRAMES // 2)
        dz2 = np.linspace(2, 2, N_FRAMES - N_FRAMES // 2)
        dz = np.concatenate((dz1, dz2))

        theta_y1 = np.linspace(0, 0, N_FRAMES // 2)
        theta_y2 = np.linspace(0, 0, N_FRAMES - N_FRAMES // 2)
        theta_y = np.concatenate((theta_y1, theta_y2))

        theta_x1 = np.linspace(0, -math.pi / 36, N_FRAMES // 2)
        theta_x2 = np.linspace(-math.pi / 36, 0, N_FRAMES - N_FRAMES // 2)
        theta_x = np.concatenate((theta_x1, theta_x2))

        theta_z = np.linspace(0, 0, N_FRAMES)
        for i in range(N_FRAMES):
            from scene.cameras import \
                Camera  # Camera(colmap_id=0, R=np.dot(eulerAnglesToRotationMatrix([theta_x[i],theta_y[i],theta_z[i]]),ref_camera.R), T=ref_camera.T + np.array([dx[i],dy[i],dz[i]]),
            # position = ref_camera.T - np.array([dx[i],dy[i],dz[i]])
            # rotation_matrix = eulerAnglesToRotationMatrix([theta_x[i],theta_y[i],theta_z[i]])
            # transed_rotation = np.dot(rotation_matrix, ref_camera.R)
            position = ref_camera.T - np.array([dx[i], dy[i], dz[i]])
            rotation_matrix = eulerAnglesToRotationMatrix([theta_x[i], theta_y[i], theta_z[i]])
            transed_rotation = np.dot(rotation_matrix, ref_camera.R)
            eval_cameras.append(Camera(colmap_id=0, R=transed_rotation, T=position,
                                       FoVx=ref_camera.FoVx, FoVy=ref_camera.FoVy,
                                       image=ref_camera.original_image, gt_alpha_mask=None,
                                       image_name='val_{}'.format(i), uid=0, data_device='cuda'))
        return eval_cameras
    elif scene_name == 'brandenburg_gate' and seq==1:
        N_FRAMES = 300
        eval_cameras = []
        ref_camera = scene.getTestCameras()[1]#5 excel line 25
        print(ref_camera.colmap_id)
        print(ref_camera.R)
        print(ref_camera.T)


        dx1 = np.linspace(0., 0., N_FRAMES//2)
        dx2 = np.linspace(0, -4, N_FRAMES - N_FRAMES // 2)
        dx = np.concatenate((dx1, dx2))

        dy1 = np.linspace(0., 0, N_FRAMES // 2)
        dy2 = np.linspace(0, -1, N_FRAMES - N_FRAMES // 2)
        dy = np.concatenate((dy1, dy2))

        dz1 = np.linspace(0, 3.5, N_FRAMES // 2)
        dz2 = np.linspace(3.5, -1.5, N_FRAMES - N_FRAMES // 2)
        dz = np.concatenate((dz1, dz2))

        theta_y1 = np.linspace(0 , 0, N_FRAMES // 2)
        theta_y2 = np.linspace(0, math.pi/6, N_FRAMES - N_FRAMES // 2)
        theta_y = np.concatenate((theta_y1, theta_y2))

        theta_x1 = np.linspace(0 , 0, N_FRAMES // 2)
        theta_x2 = np.linspace(0, -math.pi/18, N_FRAMES - N_FRAMES // 2)
        theta_x = np.concatenate((theta_x1, theta_x2))

        theta_z = np.linspace(0, 0, N_FRAMES)
        for i in range(N_FRAMES):
            from scene.cameras import Camera #Camera(colmap_id=0, R=np.dot(eulerAnglesToRotationMatrix([theta_x[i],theta_y[i],theta_z[i]]),ref_camera.R), T=ref_camera.T + np.array([dx[i],dy[i],dz[i]]),
            # position = ref_camera.T - np.array([dx[i],dy[i],dz[i]])
            # rotation_matrix = eulerAnglesToRotationMatrix([theta_x[i],theta_y[i],theta_z[i]])
            # transed_rotation = np.dot(rotation_matrix, ref_camera.R)
            position = ref_camera.T - np.array([dx[i],dy[i],dz[i]])
            rotation_matrix = eulerAnglesToRotationMatrix([theta_x[i],theta_y[i],theta_z[i]])
            transed_rotation = np.dot(rotation_matrix, ref_camera.R)
            eval_cameras.append(Camera(colmap_id=0, R=transed_rotation, T=position,
                                       FoVx=ref_camera.FoVx, FoVy=ref_camera.FoVy,
                                       image=ref_camera.original_image, gt_alpha_mask=None,
                                       image_name='val_{}'.format(i), uid=0, data_device='cuda'))
        return eval_cameras
    elif scene_name == 'brandenburg_gate' and seq == 2:
        N_FRAMES = 300
        eval_cameras = []
        ref_camera = scene.getTestCameras()[1]  # 5 excel line 25
        print(ref_camera.colmap_id)
        print(ref_camera.R)
        print(ref_camera.T)

        dx1 = np.linspace(-3., 0., N_FRAMES // 2)
        dx2 = np.linspace(0, 3, N_FRAMES - N_FRAMES // 2)
        dx = np.concatenate((dx1, dx2))

        dy1 = np.linspace(0., 0, N_FRAMES // 2)
        dy2 = np.linspace(0, 0, N_FRAMES - N_FRAMES // 2)
        dy = np.concatenate((dy1, dy2))

        dz1 = np.linspace(0, 0, N_FRAMES // 2)
        dz2 = np.linspace(0,0 , N_FRAMES - N_FRAMES // 2)
        dz = np.concatenate((dz1, dz2))

        theta_y1 = np.linspace(math.pi/9, 0, N_FRAMES // 2)
        theta_y2 = np.linspace(0, -math.pi/9, N_FRAMES - N_FRAMES // 2)
        theta_y = np.concatenate((theta_y1, theta_y2))

        theta_x1 = np.linspace(0, 0, N_FRAMES // 2)
        theta_x2 = np.linspace(0, 0, N_FRAMES - N_FRAMES // 2)
        theta_x = np.concatenate((theta_x1, theta_x2))

        theta_z = np.linspace(0, 0, N_FRAMES)
        for i in range(N_FRAMES):
            from scene.cameras import \
                Camera  # Camera(colmap_id=0, R=np.dot(eulerAnglesToRotationMatrix([theta_x[i],theta_y[i],theta_z[i]]),ref_camera.R), T=ref_camera.T + np.array([dx[i],dy[i],dz[i]]),
            # position = ref_camera.T - np.array([dx[i],dy[i],dz[i]])
            # rotation_matrix = eulerAnglesToRotationMatrix([theta_x[i],theta_y[i],theta_z[i]])
            # transed_rotation = np.dot(rotation_matrix, ref_camera.R)
            position = ref_camera.T - np.array([dx[i], dy[i], dz[i]])
            rotation_matrix = eulerAnglesToRotationMatrix([theta_x[i], theta_y[i], theta_z[i]])
            transed_rotation = np.dot(rotation_matrix, ref_camera.R)
            eval_cameras.append(Camera(colmap_id=0, R=transed_rotation, T=position,
                                       FoVx=ref_camera.FoVx, FoVy=ref_camera.FoVy,
                                       image=ref_camera.original_image, gt_alpha_mask=None,
                                       image_name='val_{}'.format(i), uid=0, data_device='cuda'))
        return eval_cameras
    elif scene_name == 'sacre_coeur' and seq==1:
        N_FRAMES = 300

        eval_cameras = []
        ref_camera = scene.getTestCameras()[13]#23
        print(ref_camera.colmap_id)
        print(ref_camera.R)
        print(ref_camera.T)

        torchvision.utils.save_image(ref_camera.original_image,'/home/wangyz/Downloads/7.png')
        dx1 = np.linspace(-2.5, 0., N_FRAMES//2)
        dx2 = np.linspace(0, 0, N_FRAMES - N_FRAMES // 2)
        dx = np.concatenate((dx1, dx2))

        dy1 = np.linspace(0, -0.5, N_FRAMES // 2)
        dy2 = np.linspace(-0.5,-0.3, N_FRAMES - N_FRAMES // 2)
        dy = np.concatenate((dy1, dy2))


        dz1 = np.linspace(0.7, 1.5, N_FRAMES // 2)
        dz2 = np.linspace(1.5, 0.5, N_FRAMES - N_FRAMES // 2)
        dz = np.concatenate((dz1, dz2))

        theta_x1 = np.linspace(-math.pi/18, 0, N_FRAMES // 2)
        theta_x2 = np.linspace(0,-math.pi/18, N_FRAMES - N_FRAMES // 2)
        theta_x = np.concatenate((theta_x1, theta_x2))
        theta_y1 = np.linspace(math.pi/6, 0, N_FRAMES // 2)
        theta_y2 = np.linspace(0, 0, N_FRAMES - N_FRAMES // 2)
        theta_y = np.concatenate((theta_y1, theta_y2))

        theta_z = np.linspace(0, 0, N_FRAMES)
        for i in range(N_FRAMES):

            from scene.cameras import Camera #Camera(colmap_id=0, R=np.dot(eulerAnglesToRotationMatrix([theta_x[i],theta_y[i],theta_z[i]]),ref_camera.R), T=ref_camera.T + np.array([dx[i],dy[i],dz[i]]),
            position = ref_camera.T - np.array([dx[i],dy[i],dz[i]])
            rotation_matrix = eulerAnglesToRotationMatrix([theta_x[i],theta_y[i],theta_z[i]])
            transed_rotation = np.dot(rotation_matrix, ref_camera.R)
            eval_cameras.append(Camera(colmap_id=0, R=transed_rotation, T=position,
                                       FoVx=ref_camera.FoVx, FoVy=ref_camera.FoVy,
                                       image=ref_camera.original_image, gt_alpha_mask=None,
                                       image_name='val_{}'.format(i), uid=0, data_device='cuda'))
        return eval_cameras
    elif scene_name == 'sacre_coeur' and seq==2:
        N_FRAMES = 600

        eval_cameras = []
        ref_camera = scene.getTestCameras()[17]#23
        print(ref_camera.colmap_id)
        print(ref_camera.R)
        print(ref_camera.T)

        torchvision.utils.save_image(ref_camera.original_image,'/home/wangyz/Downloads/7.png')
        dx1 = np.linspace(0, 0., N_FRAMES//2)
        dx2 = np.linspace(0, 0, N_FRAMES - N_FRAMES // 2)
        dx = np.concatenate((dx1, dx2))

        dy1 = np.linspace(0, 0, N_FRAMES // 2)
        dy2 = np.linspace(0,0, N_FRAMES - N_FRAMES // 2)
        dy = np.concatenate((dy1, dy2))


        dz1 = np.linspace(0, 3, N_FRAMES // 2)
        dz2 = np.linspace(3, 6, N_FRAMES - N_FRAMES // 2)
        dz = np.concatenate((dz1, dz2))

        theta_x1 = np.linspace(0, 0, N_FRAMES // 2)
        theta_x2 = np.linspace(0,-math.pi/36, N_FRAMES - N_FRAMES // 2)
        theta_x = np.concatenate((theta_x1, theta_x2))
        theta_y1 = np.linspace(0, 0, N_FRAMES // 2)
        theta_y2 = np.linspace(0, 0, N_FRAMES - N_FRAMES // 2)
        theta_y = np.concatenate((theta_y1, theta_y2))

        theta_z = np.linspace(0, 0, N_FRAMES)
        for i in range(N_FRAMES):

            from scene.cameras import Camera #Camera(colmap_id=0, R=np.dot(eulerAnglesToRotationMatrix([theta_x[i],theta_y[i],theta_z[i]]),ref_camera.R), T=ref_camera.T + np.array([dx[i],dy[i],dz[i]]),
            position = ref_camera.T - np.array([dx[i],dy[i],dz[i]])
            rotation_matrix = eulerAnglesToRotationMatrix([theta_x[i],theta_y[i],theta_z[i]])
            transed_rotation = np.dot(rotation_matrix, ref_camera.R)
            eval_cameras.append(Camera(colmap_id=0, R=transed_rotation, T=position,
                                       FoVx=ref_camera.FoVx, FoVy=ref_camera.FoVy,
                                       image=ref_camera.original_image, gt_alpha_mask=None,
                                       image_name='val_{}'.format(i), uid=0, data_device='cuda'))
        return eval_cameras
    elif scene_name == 'trevi_fountain' and seq==1:
            N_FRAMES = 600
            eval_cameras = []
            ref_camera = scene.getTestCameras()[11]  # 5
            print(ref_camera.colmap_id)
            print(ref_camera.R)
            print(ref_camera.T)


            dx1 = np.linspace(-0., 0., N_FRAMES // 2)
            dx2 = np.linspace(0, 0., N_FRAMES - N_FRAMES // 2)
            dx = np.concatenate((dx1, dx2))

            dy1 = np.linspace(-0., 0, N_FRAMES // 2)
            dy2 = np.linspace(0, -0.8, N_FRAMES - N_FRAMES // 2)
            dy = np.concatenate((dy1, dy2))

            dz1 = np.linspace(0, 3, N_FRAMES // 2)
            dz2 = np.linspace(3, 7 , N_FRAMES - N_FRAMES // 2)
            dz = np.concatenate((dz1, dz2))

            theta_y1 = np.linspace(0, 0, N_FRAMES // 2)
            theta_y2 = np.linspace(0, 0, N_FRAMES - N_FRAMES // 2)
            theta_y = np.concatenate((theta_y1, theta_y2))
            theta_x = np.linspace(0, math.pi/36, N_FRAMES)
            theta_z = np.linspace(0, 0, N_FRAMES)
            for i in range(N_FRAMES):
                from scene.cameras import \
                    Camera  # Camera(colmap_id=0, R=np.dot(eulerAnglesToRotationMatrix([theta_x[i],theta_y[i],theta_z[i]]),ref_camera.R), T=ref_camera.T + np.array([dx[i],dy[i],dz[i]]),
                position = ref_camera.T - np.array([dx[i], dy[i], dz[i]])
                rotation_matrix = eulerAnglesToRotationMatrix([theta_x[i], theta_y[i], theta_z[i]])
                transed_rotation = np.dot(rotation_matrix, ref_camera.R)

                eval_cameras.append(Camera(colmap_id=0, R=transed_rotation, T=position,
                                           FoVx=ref_camera.FoVx, FoVy=ref_camera.FoVy,
                                           image=ref_camera.original_image, gt_alpha_mask=None,
                                           image_name='val_{}'.format(i), uid=0, data_device='cuda'))
            return eval_cameras
    elif scene_name == 'trevi_fountain' and seq == 2:
        N_FRAMES = 300
        eval_cameras = []
        ref_camera = scene.getTestCameras()[17]  # 5
        print(ref_camera.colmap_id)
        print(ref_camera.R)
        print(ref_camera.T)

        dx1 = np.linspace(-2., 0., N_FRAMES // 2)
        dx2 = np.linspace(0, 2., N_FRAMES - N_FRAMES // 2)
        dx = np.concatenate((dx1, dx2))

        dy1 = np.linspace(-0., 0, N_FRAMES // 2)
        dy2 = np.linspace(0, 0, N_FRAMES - N_FRAMES // 2)
        dy = np.concatenate((dy1, dy2))

        dz1 = np.linspace(0, 0, N_FRAMES // 2)
        dz2 = np.linspace(0, 0, N_FRAMES - N_FRAMES // 2)
        dz = np.concatenate((dz1, dz2))

        theta_y1 = np.linspace(+math.pi/6, 0, N_FRAMES // 2)
        theta_y2 = np.linspace(0, -math.pi/6, N_FRAMES - N_FRAMES // 2)
        theta_y = np.concatenate((theta_y1, theta_y2))
        theta_x = np.linspace(0, 0, N_FRAMES)
        theta_z = np.linspace(0, 0, N_FRAMES)
        for i in range(N_FRAMES):
            from scene.cameras import \
                Camera  # Camera(colmap_id=0, R=np.dot(eulerAnglesToRotationMatrix([theta_x[i],theta_y[i],theta_z[i]]),ref_camera.R), T=ref_camera.T + np.array([dx[i],dy[i],dz[i]]),
            position = ref_camera.T - np.array([dx[i], dy[i], dz[i]])
            rotation_matrix = eulerAnglesToRotationMatrix([theta_x[i], theta_y[i], theta_z[i]])
            transed_rotation = np.dot(rotation_matrix, ref_camera.R)

            eval_cameras.append(Camera(colmap_id=0, R=transed_rotation, T=position,
                                       FoVx=ref_camera.FoVx, FoVy=ref_camera.FoVy,
                                       image=ref_camera.original_image, gt_alpha_mask=None,
                                       image_name='val_{}'.format(i), uid=0, data_device='cuda'))
        return eval_cameras
    elif scene_name == 'trevi_fountain' and seq == 3:
        N_FRAMES = 100
        eval_cameras = []
        ref_camera = scene.getTestCameras()[17]  # 5
        print(ref_camera.colmap_id)
        print(ref_camera.R)
        print(ref_camera.T)
        dx1 = np.linspace(-2., -2., N_FRAMES+1)
        dy1 = np.linspace(-0., 0, N_FRAMES+1)
        dz1 = np.linspace(0, 0, N_FRAMES+1)
        theta_x1 = np.linspace(0, 0, N_FRAMES+1 )
        theta_y1 = np.linspace(+math.pi/6, +math.pi/6, N_FRAMES+1)
        theta_z1 = np.linspace(0, 0, N_FRAMES+1)

        dx2 = np.linspace(-2., 0, N_FRAMES+1)
        dy2 = np.linspace(0., 0, N_FRAMES+1)
        dz2 = np.linspace(0, 0, N_FRAMES+1)
        theta_x2 = np.linspace(0, 0, N_FRAMES+1)
        theta_y2 = np.linspace(+math.pi/6, +math.pi/12, N_FRAMES+1)
        theta_z2 = np.linspace(0, 0, N_FRAMES+1)

        dx3 = np.linspace(0., -1.2, N_FRAMES+1)
        dy3 = np.linspace(0., 0, N_FRAMES+1)
        dz3 = np.linspace(0, 4, N_FRAMES+1)
        theta_x3 = np.linspace(0, math.pi/12, N_FRAMES+1)
        theta_y3 = np.linspace(+math.pi/12, -math.pi/36, N_FRAMES+1)
        theta_z3 = np.linspace(0, 0, N_FRAMES+1)

        dx4 = np.linspace(-1.2, 0, N_FRAMES+1)
        dy4 = np.linspace(0., 0, N_FRAMES+1)
        dz4 = np.linspace(4, 0, N_FRAMES+1)
        theta_x4 = np.linspace(math.pi/12, 0, N_FRAMES+1)
        theta_y4 = np.linspace(-math.pi/36, 0, N_FRAMES+1)
        theta_z4 = np.linspace(0, 0, N_FRAMES+1)

        dx5 = np.linspace(0, 0, N_FRAMES+1)
        dy5 = np.linspace(0., 0, N_FRAMES+1)
        dz5 = np.linspace(0, 4, N_FRAMES+1)
        theta_x5 = np.linspace(0, math.pi/36, N_FRAMES+1)
        theta_y5 = np.linspace(0, 0, N_FRAMES+1)
        theta_z5 = np.linspace(0, 0, N_FRAMES+1)



        dx6 = np.linspace(0, 0, N_FRAMES+1)
        dy6 = np.linspace(0., 0, N_FRAMES+1)
        dz6 = np.linspace(4, 4, N_FRAMES+1)
        theta_x6 = np.linspace(math.pi/36, math.pi/36, N_FRAMES+1)
        theta_y6 = np.linspace(0, 0, N_FRAMES+1)
        theta_z6 = np.linspace(0, 0, N_FRAMES+1)

        dx7 = np.linspace(0, 0, N_FRAMES+1)
        dy7 = np.linspace(0., 0, N_FRAMES+1)
        dz7 = np.linspace(4, 0, N_FRAMES+1)
        theta_x7 = np.linspace(math.pi/36, 0, N_FRAMES+1)
        theta_y7 = np.linspace(0, 0, N_FRAMES+1)
        theta_z7 = np.linspace(0, 0, N_FRAMES+1)

        dx8 = np.linspace(0, 1, N_FRAMES*2+2)
        dy8 = np.linspace(0., 0, N_FRAMES*2+2)
        dz8 = np.linspace(0, 4, N_FRAMES*2+2)
        theta_x8 = np.linspace(0, 0, N_FRAMES*2+2)
        theta_y8 = np.linspace(0,+math.pi/24 , N_FRAMES*2+2)
        theta_z8 = np.linspace(0, 0, N_FRAMES*2+2)


        dx = np.concatenate((dx1, dx2,dx3,dx4,dx5,dx6,dx7,dx8))
        dy = np.concatenate((dy1, dy2,dy3,dy4,dy5,dy6,dy7,dy8))
        dz = np.concatenate((dz1, dz2,dz3,dz4,dz5,dz6,dz7,dz8))
        theta_x = np.concatenate((theta_x1, theta_x2,theta_x3,theta_x4,theta_x5,theta_x6,theta_x7,theta_x8))
        theta_y = np.concatenate((theta_y1, theta_y2,theta_y3,theta_y4,theta_y5,theta_y6,theta_y7,theta_y8))
        theta_z = np.concatenate((theta_z1, theta_z2,theta_z3,theta_z4,theta_z5,theta_z6,theta_z7,theta_z8))
        # dx1 = np.linspace(-3., 0., N_FRAMES // 2)
        # dx2 = np.linspace(0, 3., N_FRAMES - N_FRAMES // 2)
        # dx = np.concatenate((dx1, dx2))
        #
        # dy1 = np.linspace(-0., 0, N_FRAMES // 2)
        # dy2 = np.linspace(0, 0, N_FRAMES - N_FRAMES // 2)
        # dy = np.concatenate((dy1, dy2))
        #
        # dz1 = np.linspace(0, 0, N_FRAMES // 2)
        # dz2 = np.linspace(0, 0, N_FRAMES - N_FRAMES // 2)
        # dz = np.concatenate((dz1, dz2))
        #
        # theta_y1 = np.linspace(+math.pi/6, 0, N_FRAMES // 2)
        # theta_y2 = np.linspace(0, -math.pi/6, N_FRAMES - N_FRAMES // 2)
        # theta_y = np.concatenate((theta_y1, theta_y2))
        # theta_x = np.linspace(0, 0, N_FRAMES)
        # theta_z = np.linspace(0, 0, N_FRAMES)
        for i in range(903):
            from scene.cameras import \
                Camera  # Camera(colmap_id=0, R=np.dot(eulerAnglesToRotationMatrix([theta_x[i],theta_y[i],theta_z[i]]),ref_camera.R), T=ref_camera.T + np.array([dx[i],dy[i],dz[i]]),
            position = ref_camera.T - np.array([dx[i], dy[i], dz[i]])
            rotation_matrix = eulerAnglesToRotationMatrix([theta_x[i], theta_y[i], theta_z[i]])
            transed_rotation = np.dot(rotation_matrix, ref_camera.R)

            eval_cameras.append(Camera(colmap_id=0, R=transed_rotation, T=position,
                                       FoVx=ref_camera.FoVx, FoVy=ref_camera.FoVy,
                                       image=ref_camera.original_image, gt_alpha_mask=None,
                                       image_name='val_{}'.format(i), uid=0, data_device='cuda'))
        return eval_cameras
    else:
        raise NotImplementedError

def render_set(model_path, source_path, name, iteration, views, gaussians, pipeline, background, args,which_feature_fusion_func,num_aug_rendering):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    render_npy_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_npy")
    gts_npy_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt_npy")

    makedirs(render_npy_path, exist_ok=True)
    makedirs(gts_npy_path, exist_ok=True)
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        output = render(view, gaussians, pipeline, background, args)

        if not args.include_feature:
            rendering = output["render"]
        else:
            rendering = output["language_feature_image"]
            
        if not args.include_feature:
            gt = view.original_image[0:3, :, :]
            
        else:
            if name == 'eval':
                if_eval = True
            else:
                if_eval = False
            gt, mask_t,mask_a = view.get_language_feature(os.path.join(source_path, args.language_features_name), args.feature_level,which_feature_fusion_func,num_aug_rendering,eval=if_eval)

        # PT style
        np.save(os.path.join(render_npy_path, view.image_name + ".npy"),rendering.permute(1,2,0).cpu().numpy())
        if gt is not None:
            np.save(os.path.join(gts_npy_path, view.image_name + ".npy"),gt.permute(1,2,0).cpu().numpy())
        if rendering.shape[0]>3:
            torchvision.utils.save_image(rendering[:3], os.path.join(render_path, view.image_name + "1.png"))
            torchvision.utils.save_image(rendering[3:6], os.path.join(render_path, view.image_name + "2.png"))
            torchvision.utils.save_image(rendering[6:9], os.path.join(render_path, view.image_name + "3.png"))
            torchvision.utils.save_image(rendering[9:12], os.path.join(render_path, view.image_name + "4.png"))
            if gt is not None:
                torchvision.utils.save_image(gt[:3], os.path.join(gts_path, view.image_name + ".png"))
        else:
            torchvision.utils.save_image(rendering, os.path.join(render_path, view.image_name + ".png"))
            if gt is not None:
                torchvision.utils.save_image(gt, os.path.join(gts_path, view.image_name + ".png"))


        # np.save(os.path.join(render_npy_path, '{0:05d}'.format(idx) + ".npy"),rendering.permute(1,2,0).cpu().numpy())
        # np.save(os.path.join(gts_npy_path, '{0:05d}'.format(idx) + ".npy"),gt.permute(1,2,0).cpu().numpy())
        # torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        # torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool,skip_eval : bool, args):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree,dataset.num_aug_rendering)
        scene = Scene(dataset, gaussians, shuffle=False)
        checkpoint = os.path.join(args.model_path, 'chkpnt30000.pth')
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, args, mode='test')
        
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             print('render training split...')
             if args.render_small_batch == True:
                 train_cams = scene.getTrainCameras()
                 scene_name = dataset.source_path.split('/')[-1]
                 gt_json_folder = os.listdir(os.path.join(dataset.source_path.split('PT')[0],'PT','label',scene_name))
                 img_list = [x for x in gt_json_folder if x.endswith('.jpg')]
                 cams = []
                 for i in range(len(train_cams)):
                     this_cam = train_cams[i]
                     if this_cam.image_name+'.jpg' in img_list:
                         cams.append(this_cam)
             else:
                 cams = scene.getTrainCameras()
             render_set(dataset.model_path, dataset.source_path, "train", scene.loaded_iter, cams, gaussians, pipeline, background, args,dataset.which_feature_fusion_func,dataset.num_aug_rendering)

        if not skip_test:
             print('render test split...')
             render_set(dataset.model_path, dataset.source_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, args,dataset.which_feature_fusion_func,dataset.num_aug_rendering)
        if not skip_eval:
            print('render eval split...')
            eval_cameras = constructEvalCameras(scene)
            render_set(dataset.model_path, dataset.source_path, "eval", scene.loaded_iter, eval_cameras, gaussians, pipeline, background, args,dataset.which_feature_fusion_func,dataset.num_aug_rendering)
if __name__ == "__main__":
    # Set up command line argument parser
    
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--skip_eval",action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--include_feature", action="store_true")
    parser.add_argument("--render_small_batch",action="store_true")


    args = get_combined_args(parser)
    if args.which_feature_fusion_func == 'default':
        if args.num_aug_rendering != 0:
            print('Warning: number of augmentation rendering is not 0.SET: num_aug_rendering = 0')
            args.num_aug_rendering = 0
    print("Rendering the LangSplat Model:" + args.model_path)

    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test,args.skip_eval, args)