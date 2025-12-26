from argparse import ArgumentParser
from re import search

from fsspec.registry import default

from arguments import ModelParams, PipelineParams, OptimizationParams
from arguments import itw_ModelParams, itw_PipelineParams, itw_AEParams, itw_TEParams, get_combined_args
from utils.general_utils import safe_state
import sys
import torch
import numpy as np
import datetime
import random
from in_the_wild_renderer.itw_gaussian_renderer import GaussianModel, gaussians_renderer
from in_the_wild_renderer.itw_scene import Scene
from in_the_wild_renderer.appearance_encoder import createAppearanceEncoder
from in_the_wild_renderer.transient_encoder import createTransientEncoder
import os
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as tf
class IntheWildGaussiansRenderer:
    def __init__(self):
        pass
    def render_with_path(self,source_image, appearance_target):
        pass
    def batch_render_with_path(self,source_images, appearance_target_list):
        # return transed images
        pass


class WEGSRenderer(IntheWildGaussiansRenderer):
    def __init__(self,dataset,appearance_encoder_para,transient_encoder_para,pipeline_para,iteration):
        super().__init__()
        with torch.no_grad():
            self.gaussians = GaussianModel(dataset.itw_sh_degree,num_aug_rendering=3)
            self.scene = Scene(dataset, self.gaussians, load_iteration=iteration, shuffle=False)
            ae = createAppearanceEncoder(appearance_encoder_para, self.scene)
            te = createTransientEncoder(transient_encoder_para, self.scene)
            ae.load_latest_checkpoint(file_path=self.scene.model_path)
            te.load_latest_checkpoint(file_path=self.scene.model_path)
            self.ae = ae.to('cuda')
            self.te = te.to('cuda')
            self.pipeline_para = pipeline_para
            bg_color = [1, 1, 1] if dataset.itw_white_background else [0, 0, 0]
            self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
            self.output_folder = os.path.join(dataset.itw_source_path, 'language_feature_mul_app')
            os.makedirs(self.output_folder, exist_ok=True)
            print("Output folder:{}".format(self.output_folder))
            self.render_view = None
            self.target_appearance_view_list = []
            self.target_rendering_result_dict = {}
            self.target_mask_result_dict = {}
            self.target_original_image_dict = {}

    def render_with_path(self,render_view,target_appearance_view):
        # search image by image name
        with torch.no_grad():
            mask,_ = self.te(target_appearance_view.original_image)
            embeddings_target = self.ae.encode(target_appearance_view.original_image, mask)
            rendering_result = gaussians_renderer(self.ae, render_view, self.gaussians,self.pipeline_para, self.background, self_embeddings = embeddings_target)["render"]
        return rendering_result, mask
    def batch_render_with_path(self,render_cam_view_path : str, appearance_target_path_list : list[str]):
        with torch.no_grad():
            # search image by image name
            views = self.scene.getTrainCameras()
            search_img_cnt = 0
            for idx in range(len(views)):
                if views[idx].image_name == render_cam_view_path:
                    self.render_view = views[idx]
                    search_img_cnt += 1
                if views[idx].image_name in appearance_target_path_list:
                    self.target_appearance_view_list.append(views[idx])
                    search_img_cnt += 1
            assert search_img_cnt == len(appearance_target_path_list) + 1, 'Error, some image path not in either training nor testing data'
            for target_appearance_view in self.target_appearance_view_list:
                # torchvision.utils.save_image(target_appearance_view.original_image,os.path.join(self.output_folder, "target_appearance_{}_GT.png".format(target_appearance_view.image_name)))
                rendering_result, mask  = self.render_with_path(self.render_view, target_appearance_view)
                self.target_rendering_result_dict[target_appearance_view.image_name] = rendering_result
                self.target_mask_result_dict[target_appearance_view.image_name] = mask
                self.target_original_image_dict[target_appearance_view.image_name] = target_appearance_view.original_image
                # torchvision.utils.save_image(rendering_result,os.path.join(self.output_folder, "target_appearance_{}_render.png".format(target_appearance_view.image_name)))
                # torchvision.utils.save_image(mask,os.path.join(self.output_folder, "target_appearance_{}_mask.png".format(target_appearance_view.image_name)))
        return self.target_rendering_result_dict, self.target_mask_result_dict
    def batch_render_and_save_all_training_data(self, appearance_target_path_list,flag_save_rendering_image=True):
        with torch.no_grad():
            views = self.scene.getTrainCameras()
            search_img_cnt = 0
            for idx in range(len(views)):
                if views[idx].image_name in appearance_target_path_list:
                    self.target_appearance_view_list.append(views[idx])
                    search_img_cnt += 1
            assert search_img_cnt == len (appearance_target_path_list), 'Error, some image path not in the training or testing dataset'
            self.novel_app_aug_dict = {}  # {'view_name':[view_1_render,view_2_render,view_3_render]}
            for view in views:
                self.render_view = view
                ma_aug_list = []
                for target_appearance_view in self.target_appearance_view_list:
                    rendering_result, mask = self.render_with_path(view, target_appearance_view)
                    self.target_rendering_result_dict[target_appearance_view.image_name] = rendering_result
                    self.target_mask_result_dict[target_appearance_view.image_name] = mask
                    self.target_original_image_dict[target_appearance_view.image_name] = target_appearance_view.original_image
                    ma_aug_list.append(rendering_result)
                    if flag_save_rendering_image:
                        self.save_everything()
                stacked_ma_aug_list = torch.stack(ma_aug_list)
                self.novel_app_aug_dict[view.image_name] = stacked_ma_aug_list.cpu().numpy()
                self.clear_everything()
        return self.novel_app_aug_dict
    def batch_render_and_save_all_training_data_with_self_appearance(self, appearance_target_path_list,flag_save_rendering_image=True):
        print(len (appearance_target_path_list))
        with torch.no_grad():
            views = self.scene.getTrainCameras()
            search_img_cnt = 0
            # FIXME： OCCOR A BUG FOR ON-THE-GO DATASET， DOUBLE VIEWS WILL BE SEARCHED
            searched_buf = []
            for idx in range(len(views)):
                if views[idx].image_name in appearance_target_path_list:
                    if views[idx].image_name not in searched_buf:
                        searched_buf.append(views[idx].image_name)
                        self.target_appearance_view_list.append(views[idx])
                        search_img_cnt += 1

            assert search_img_cnt == len (appearance_target_path_list), 'Error, some image path not in the training or testing dataset'
            self.novel_app_aug_dict = {}  # {'view_name':[view_1_render,view_2_render,view_3_render]}
            for view in views:
                self.render_view = view
                ma_aug_list = []
                for target_appearance_view in self.target_appearance_view_list:
                    rendering_result, mask = self.render_with_path(view, target_appearance_view)
                    self.target_rendering_result_dict[target_appearance_view.image_name] = rendering_result
                    self.target_mask_result_dict[target_appearance_view.image_name] = mask
                    self.target_original_image_dict[target_appearance_view.image_name] = target_appearance_view.original_image
                    ma_aug_list.append(rendering_result)
                # render with self_appearance
                rendering_result, mask = self.render_with_path(view, view)
                self.target_rendering_result_dict[view.image_name] = rendering_result
                self.target_mask_result_dict[view.image_name] = mask
                self.target_original_image_dict[view.image_name] = view.original_image
                ma_aug_list.append(rendering_result)
                if flag_save_rendering_image:
                    self.save_everything()
                stacked_ma_aug_list = torch.stack(ma_aug_list)
                self.novel_app_aug_dict[view.image_name] = stacked_ma_aug_list.cpu().numpy()
                self.clear_everything()
        return self.novel_app_aug_dict
    def save_everything_debug(self):
        assert self.render_view is not None, "[ERROR] render_view is None"
        torchvision.utils.save_image(self.render_view.original_image, os.path.join(self.output_folder,"render_view_{}.png".format(self.render_view.image_name)))
        for key in self.target_rendering_result_dict.keys():
            target_rendering_result = self.target_rendering_result_dict[key]
            target_mask_result = self.target_mask_result_dict[key]
            target_original_image = self.target_original_image_dict[key]
            _, h, w = target_original_image.shape
            transform = transforms.Resize([h, w])
            target_mask_result = transform(target_mask_result[0].unsqueeze(0)).repeat(3, 1, 1)
            torchvision.utils.save_image(target_rendering_result, os.path.join(self.output_folder, "target_{}_render.png".format(key)))
            torchvision.utils.save_image(target_mask_result, os.path.join(self.output_folder, "target_{}_mask.png".format(key)))
            torchvision.utils.save_image(target_original_image,os.path.join(self.output_folder, "target_{}_original_image.png".format(key)))

    def save_everything(self):
        assert self.render_view is not None, "[ERROR] render_view is None"
        for idx, key in enumerate(self.target_rendering_result_dict.keys()):
            target_rendering_result = self.target_rendering_result_dict[key]
            target_mask_result = self.target_mask_result_dict[key]
            target_original_image = self.target_original_image_dict[key]
            _, h, w = target_original_image.shape
            torchvision.utils.save_image(target_rendering_result, os.path.join(self.output_folder, "{}_{}_render.png".format(self.render_view.image_name,idx)))

    def clear_everything(self):
        self.render_view = None
        self.target_rendering_result_dict = {}
        self.target_mask_result_dict = {}
        self.target_original_image_dict = {}


if __name__ == "__main__":
    seed_num = 42 # 3407 114514

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    itw_model = itw_ModelParams(parser)
    itw_ap = itw_AEParams(parser)
    itw_te = itw_TEParams(parser)
    itw_pipeline = itw_PipelineParams(parser)
    parser.add_argument("--iteration", default=200000, type=int)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=55555)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--render_cam_view_path", type=str, default = "76879175_4725301545")
    parser.add_argument("--appearance_target_path_list", type=str, default=['15080601_1551250100','01738801_5114523193','59826471_8014732885'])
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    print(args)
    args.model_path = args.model_path + f"_{str(args.feature_level)}"
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    itw_mp = itw_model.extract(args)
    itw_ap = itw_ap.extract(args)
    itw_te = itw_te.extract(args)
    itw_pipeline = itw_pipeline.extract(args)
    print("Optimizing " + args.model_path)
    itw_renderer = WEGSRenderer(itw_mp,itw_ap,itw_te,itw_pipeline,args.iteration)
    renderings, masks = itw_renderer.batch_render_and_save_all_training_data(args.appearance_target_path_list)
