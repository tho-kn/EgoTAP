import argparse
import os
from utils import util
import torch
from .base_options import BaseOptions
import pathlib

class DatasetOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)

        self.parser.add_argument('--default_data_path', type=str, default="./UnrealEgoData",
                                help='default path to the dataset written in data list')
        self.parser.add_argument('--data_dir', type=str, default="/ssd_data1/UnrealEgoData",
                                help='path to the dataset')
        self.parser.add_argument('--data_sub_path', type=str, default='all_data_with_img-256_hm-64_pose-16_npy',
                                help='sub path to npy files')
        self.parser.add_argument('--metadata_dir', nargs='+', type=str,
                                default=[str(pathlib.Path.home())+"/nas/UnrealEgoData"],
                                help='default path to the dataset metadata files')
        self.parser.add_argument('--data_prefix', type=str, default="",
                                help='prefix to the dataset list files')
        self.parser.add_argument('--joint_preset', type=str, default="UnrealEgo",
                                 help='preset for joint order and parents')
        
    def parse(self, custom_args=None):
        super().parse(custom_args)
        
        # Set options based on dataset
        self.opt.estimate_head = False
        self.opt.stereo = True
        
        if self.opt.joint_preset == "UnrealEgo":
            self.opt.estimate_head = True
            self.opt.stereo = True
        if self.opt.joint_preset == "EgoCap":
            self.opt.estimate_head = False
            self.opt.stereo = True
        if self.opt.joint_preset == "xR-Egopose":
            self.opt.estimate_head = True
            self.opt.stereo = False
        
        return self.opt
        