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

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = "" 
        self._language_features_name = "language_features_dim3_VanillaCLIPExt"
        self._images = "images"
        self._resolution = 8
        self._white_background = False
        self._feature_level = -1
        self.which_feature_fusion_func = "post_validMapLevel_avgImageWiseMaxValue|LocMax" #choices:default, aug;
        self.num_aug_rendering = 3 # ,must set=0 with the vanilla LangSPalt
        self.data_device = "cuda"
        self.eval = False
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        g.lf_path = os.path.join(g.source_path, g.language_features_name)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.language_feature_lr = 0.0025 # TODO: update
        self.include_feature = True # Set to False if train the original gs
        self.beta_reg_diffapp = 0
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        super().__init__(parser, "Optimization Parameters")

# parameters for in-the-wild renderer

class itw_ModelParams(ParamGroup):
    def __init__(self, parser):
        self.itw_sh_degree = 5
        self.itw_evalutate_with_gtmask=1 # 1,0
        self.itw_source_path = "/media/wangyz/DATA/UBUNTU_data/dataset/PT/brandenburg_gate_tiny"
        self.itw_model_path = "/media/wangyz/DATA/UBUNTU_data/dataset/PT/brandenburg_gate/brandenburg_gate_wegs"
        self.itw_images = "images"
        self.itw_resolution = 8
        self.itw_white_background = False
        self.itw_data_device = "cuda"
        self.itw_eval = False
        super().__init__(parser, "in-the-wild renderer Pipeline Parameters")
        # self.mask_scale = 10.0


    # def extract(self, args):
    #     g = super().extract(args)
    #     g.itw_source_path = os.path.abspath(g.itw_source_path)
    #     return g

class itw_PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.itw_convert_SHs_python = False
        self.itw_compute_cov3D_python = False
        self.itw_debug = False
        super().__init__(parser, "Pipeline Parameters")


class itw_AEParams(ParamGroup):
    def __init__(self, parser):
        self.enc_type = "cnn" #glo
        self.appearance_dim = 32
        self.encode_with_mask = 1
        self.mask_func = 'cat' # cat or multiply
        self.app_pe_freq = 0
        self.pe_freq_xyz = 4
        self.transnet_lr = 1e-4
        self.encodenet_lr = 1e-4
        super().__init__(parser, "Appearance Encoder Parameters")

class itw_TEParams(ParamGroup):
    def __init__(self, parser):
        self.te_enc_type = "unet" #cnn,glo
        self.trans_uv_pe_freq = 4
        self.transient_dim = 16
        self.mask_net_lr = 1e-5
        self.te_encodenet_lr = 1e-4
        super().__init__(parser, "Transient Encoder Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)

