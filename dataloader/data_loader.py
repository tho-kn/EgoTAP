import os
import numpy as np
import torch
import torchvision.transforms as transforms
from dataloader.image_folder import make_dataset
from model.network import get_limb_dim

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        assert (isinstance(image, np.ndarray))
        image -= self.mean
        image /= self.std

        return image
    
    
def dataset_full(opt, mode='train', id=None):
    if opt.model == "egoglass":
        datasets = CreateStereoFullDataset(opt, mode, id=id)
    elif opt.model == "unrealego_autoencoder":
        datasets = CreateStereoFullDataset(opt, mode, id=id)
    elif opt.model == "heatmap_shared":
        datasets = CreateStereoFullDataset(opt, mode, id=id)
    elif opt.model == "ego3dpose_autoencoder":
        datasets = CreateStereoFullDataset(opt, mode, id=id)
    elif opt.model == "egotap_autoencoder":
        datasets = CreateStereoFullDataset(opt, mode, id=id)
    else:
        raise Exception("Undefined model is chosen")
        
    return datasets


def dataloader_full(opt, mode='train', id=None):
    if mode == 'train':
        shuffle = True
        drop_last = True
    elif mode == 'validation':
        shuffle = False
        drop_last = False
    elif mode == 'test':
        shuffle = False
        drop_last = False
    else:
        raise Exception("Undefined mode is chosen for dataloader")
        
    datasets = dataset_full(opt, mode, id)

    dataset = torch.utils.data.DataLoader(
        datasets, 
        batch_size=opt.batch_size, 
        shuffle=shuffle, 
        num_workers=int(opt.num_threads), 
        drop_last=drop_last
    )
    return dataset

from utils.util import get_index_to_joint_name
from utils.projection import generate_pseudo_limb_mask, coord2d_to_heatmap
from utils.data import overwrite_limb_data
import cv2

def resize_rgb(rgb, w, h):
    rgb = rgb.transpose(1, 2, 0)
    rgb = cv2.resize(rgb, (w, h))
    rgb = rgb.transpose(2, 0, 1)
    return rgb

def process_frame_data(frame_data_path, opt):
    assert opt.load_size_heatmap[0] == opt.load_size_heatmap[1], "Width and height of heatmap must be the same"

    # load each data
    frame_data = np.load(frame_data_path, allow_pickle=True)
    frame_data = frame_data.item()
    
    index_to_joint_name = get_index_to_joint_name(opt.joint_preset)
    num_joints = len(index_to_joint_name)
    
    heatmap_W = opt.load_size_heatmap[0]
    heatmap_H = opt.load_size_heatmap[1]
    
    hm_sigma = 1.0
    frame_data["gt_heatmap_left"] = coord2d_to_heatmap(frame_data['gt_camera_2d_left'][1:], res=heatmap_W, sigma=hm_sigma)
    
    if opt.stereo:
        frame_data["gt_heatmap_right"] = coord2d_to_heatmap(frame_data['gt_camera_2d_right'][1:], res=heatmap_W, sigma=hm_sigma)
    else:
        frame_data["gt_heatmap_right"] = frame_data["gt_heatmap_left"]
    
    # input should be resized to 4 * 4 of heatmaps
    frame_data["input_rgb_left"] = resize_rgb(frame_data["input_rgb_left"], heatmap_W * 4, heatmap_H * 4)
    input_rgb_left = torch.from_numpy(frame_data["input_rgb_left"]).float()
    gt_heatmap_left = torch.from_numpy(frame_data["gt_heatmap_left"]).float()
    
    if opt.stereo:
        frame_data["input_rgb_right"] = resize_rgb(frame_data["input_rgb_right"], heatmap_W * 4, heatmap_H * 4)
        input_rgb_right = torch.from_numpy(frame_data["input_rgb_right"]).float()
        gt_heatmap_right = torch.from_numpy(frame_data["gt_heatmap_right"]).float()
    else:
        input_rgb_right = input_rgb_left
        gt_heatmap_right = gt_heatmap_left
    
    # Add limb data
    pelvis_pose_left = frame_data["gt_pelvis_left"]
    pts2d_left = frame_data['gt_camera_2d_left']
    pts3d_left = frame_data['gt_local_pose'] + pelvis_pose_left[None, :]
    
    if opt.stereo:
        pelvis_pose_right = frame_data["gt_pelvis_right"]
        pts2d_right = frame_data['gt_camera_2d_right']
        pts3d_right = frame_data['gt_local_pose'] + pelvis_pose_right[None, :]
    else:
        pts2d_right = pts2d_left
        pts3d_right = pts3d_left
    
    # To save space, generate limb heatmaps on the fly
    overwrite_limb_data(frame_data, pts2d_left, pts2d_right, pts3d_left, pts3d_right, res=heatmap_W, area=heatmap_W,
                        htype='line', sigma=hm_sigma, joint_preset=opt.joint_preset, is_stereo=opt.stereo)
    
    gt_raw_limb_heatmap_left = torch.from_numpy(frame_data["gt_limb_heatmap_left"]).float() * 2
    
    if opt.stereo:
        gt_raw_limb_heatmap_right = torch.from_numpy(frame_data["gt_limb_heatmap_right"]).float() * 2
    else:
        gt_raw_limb_heatmap_right = gt_raw_limb_heatmap_left

    gt_local_pose = torch.from_numpy(frame_data["gt_local_pose"]).float()
    gt_limb_theta = torch.from_numpy(frame_data["gt_limb_theta"]).float()
        
    gt_local_rot = torch.from_numpy(frame_data["gt_local_rot"]).float()
    
    gt_limb_norm_left = torch.from_numpy(frame_data["gt_pixel_length_left"]).float()
    gt_pelvis_left = torch.from_numpy(frame_data["gt_pelvis_left"]).float()
    
    if opt.stereo:
        gt_limb_norm_right = torch.from_numpy(frame_data["gt_pixel_length_right"]).float()
        gt_pelvis_right = torch.from_numpy(frame_data["gt_pelvis_right"]).float()
    else:
        gt_limb_norm_right = gt_limb_norm_left
        gt_pelvis_right = gt_pelvis_left
    
    if opt.num_heatmap < num_joints:
        gt_heatmap_left = gt_heatmap_left[-opt.num_heatmap:]
        gt_heatmap_right = gt_heatmap_right[-opt.num_heatmap:]
        
    if opt.joint_preset == "UnrealEgo" and not opt.estimate_head:
        # Test head-relative pose estimation
        gt_local_pose = gt_local_pose + gt_pelvis_left[None, :]
        gt_pelvis_left = torch.zeros_like(gt_pelvis_left)
        gt_pelvis_right = torch.zeros_like(gt_pelvis_right)
        
    if 0 < opt.num_rot_heatmap < gt_raw_limb_heatmap_left.shape[0]:
        gt_raw_limb_heatmap_left = gt_raw_limb_heatmap_left[-opt.num_rot_heatmap:]
        gt_raw_limb_heatmap_right = gt_raw_limb_heatmap_right[-opt.num_rot_heatmap:]
        gt_limb_norm_left = gt_limb_norm_left[-opt.num_rot_heatmap:]
        gt_limb_norm_right = gt_limb_norm_right[-opt.num_rot_heatmap:]
        gt_limb_theta = gt_limb_theta[-opt.num_rot_heatmap:]
    
    base_data = {
        "frame_data_path": frame_data_path,
        'input_rgb_left': input_rgb_left, 
        'input_rgb_right': input_rgb_right,
        'gt_heatmap_left': gt_heatmap_left, 
        'gt_heatmap_right': gt_heatmap_right,
        'gt_pelvis_left': gt_pelvis_left,
        'gt_pelvis_right': gt_pelvis_right,
        'gt_limb_theta': gt_limb_theta,
        'gt_local_pose': gt_local_pose if opt.estimate_head else gt_local_pose[1:],
        'gt_local_rot': gt_local_rot,
    }
    
    if opt.model == 'egoglass':
        opt.seg_thickness = 10 if opt.joint_preset == "EgoCap" else 30
        
        pts2d_left = frame_data['gt_camera_2d_left']
        gt_segmentation_left = generate_pseudo_limb_mask(pts2d_left, res=heatmap_W * 4, joint_preset=opt.joint_preset)
        base_data["gt_segmentation_left"] = gt_segmentation_left
        
        if opt.stereo:
            pts2d_right = frame_data['gt_camera_2d_right']
            gt_segmentation_right = generate_pseudo_limb_mask(pts2d_right, res=heatmap_W * 4, joint_preset=opt.joint_preset)
            base_data["gt_segmentation_right"] = gt_segmentation_right
        else:
            base_data["gt_segmentation_right"] = gt_segmentation_left
    
    if opt.heatmap_type == 'sin':
        gt_rot_cos_heatmap_left = gt_raw_limb_heatmap_left * torch.cos(gt_limb_theta)[:, None, None]
        gt_rot_sin_heatmap_left = gt_raw_limb_heatmap_left * torch.sin(gt_limb_theta)[:, None, None]
        gt_rot_cos_heatmap_right = gt_raw_limb_heatmap_right * torch.cos(gt_limb_theta)[:, None, None]
        gt_rot_sin_heatmap_right = gt_raw_limb_heatmap_right * torch.sin(gt_limb_theta)[:, None, None]
        gt_limb_heatmap_left = torch.cat((gt_rot_cos_heatmap_left, gt_rot_sin_heatmap_left), dim=0)
        gt_limb_heatmap_right = torch.cat((gt_rot_cos_heatmap_right, gt_rot_sin_heatmap_right), dim=0)

    if opt.num_rot_heatmap > 0:
        limb_dim = get_limb_dim(opt)
        if opt.heatmap_type == 'limb':
            gt_limb_heatmap_left = gt_raw_limb_heatmap_left
            gt_limb_heatmap_right = gt_raw_limb_heatmap_right
            
        base_data['gt_limb_heatmap_left'] = gt_limb_heatmap_left
        base_data['gt_limb_heatmap_right'] = gt_limb_heatmap_right
        
        gt_limb_norm_left = torch.cat([gt_limb_norm_left] * limb_dim, dim=0)
        gt_limb_norm_right = torch.cat([gt_limb_norm_right] * limb_dim, dim=0)

        base_data['gt_plength_left'] = gt_limb_norm_left
        base_data['gt_plength_right'] = gt_limb_norm_right
    return base_data
        

class CreateStereoFullDataset(torch.utils.data.Dataset):
    def __init__(self, opt, mode, id=None):
        super(CreateStereoFullDataset, self).__init__()
        self.opt = opt
        self.mode = mode

        self.load_size_heatmap = opt.load_size_heatmap

        self.data_list_path = os.path.join(opt.data_dir, opt.data_prefix + mode + '.txt')  # CT\UnrealEgo\static00\UnrealEgoData\train.txt

        self.frame_data_paths, self.num_frame_data = make_dataset(
            opt=opt, 
            data_list_path=self.data_list_path, 
            data_sub_path=opt.data_sub_path,
            id=id
        )

    def __getitem__(self, index):
        # get paths for each data
        frame_data_path = self.frame_data_paths[index]
        base_data = process_frame_data(frame_data_path, self.opt)
            
        return base_data

    def __len__(self):
        return self.num_frame_data


if __name__ == "__main__":

    img = transforms.ToPILImage()(torch.randn(3, 224, 224))

    color_jitter = transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)
    transform = transforms.ColorJitter.get_params(
        color_jitter.brightness, color_jitter.contrast, color_jitter.saturation,
        color_jitter.hue)

    img_trans1 = transform(img)
    img_trans2 = transform(img)

    print((np.array(img_trans1) == np.array(img_trans2)).all())
