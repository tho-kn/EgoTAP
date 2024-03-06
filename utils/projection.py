import numpy as np
import torch
from skimage.draw import line_aa
from scipy.ndimage.filters import gaussian_filter
from utils.util import try_json, denormalize_ImageNet, get_kinematic_parents
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import os

# https://github.com/carlosferrazza/Python-Calibration/blob/master/Functions/ocam_functions.py
""" Reads file containing the ocamcalib parameters exported from the Matlab toolbox """
def get_ocam_model(opt=None, side='left'):
    global ocam_model
    if side not in ocam_model.keys():
        if opt is None:
            print("No opt given, using UnrealEgo's data_dir")
            data_dir = "/ssd_data1/UnrealEgo"
        else:
            data_dir = opt.data_dir
            
        json_path = os.path.join(data_dir, f"fisheye.calibration_{side}.json")
        json_data = try_json(json_path)
                
        o = {}
        o['name'] = json_data['name']
        
        o['length_pol'] = len(json_data["polynomialC2W"])
        o['pol'] = json_data["polynomialC2W"]
        
        o['length_invpol'] = len(json_data["polynomialW2C"])
        o['invpol'] = json_data["polynomialW2C"]
        
        o['xc'] = json_data["image_center"][1]
        o['yc'] = json_data["image_center"][0]
        
        o['c'] = json_data["affine"][0]
        o['d'] = json_data["affine"][1]
        o['e'] = json_data["affine"][2]
                
        o['height'] = json_data["size"][0]
        o['width'] = json_data["size"][1]
        
        o['radius'] = json_data["imageCircleRadius"]
        
        ocam_model[side] = o
        
    o = ocam_model[side]

    return o

ocam_model = {}


def cam2world(point2D, o=None):
    # This function assumes 1024 by 1024 sized images
    lib = torch if isinstance(point2D, torch.Tensor) else np

    shape = list(point2D.shape)
    shape[-1] = 3
    point3D = lib.zeros(tuple(shape))
    
    if isinstance(point2D, torch.Tensor):
        point3D = point3D.to(point2D.device)
    
    invdet = 1.0/(o['c']-o['d']*o['e'])

    xp = invdet*((point2D[..., 0]-o['xc']) - o['d']*(point2D[..., 1]-o['yc']))
    yp = invdet*(-o['e']*(point2D[..., 0]-o['xc']) + o['c']*(point2D[..., 1]-o['yc']))
    
    r = lib.linalg.norm([xp, yp], axis=0)
    
    zp = o['pol'][0]
    zp = lib.ones_like(r) * zp
    r_i = lib.ones_like(r)
    
    for i in range(1,o['length_pol']):
        r_i *= r
        zp += r_i*o['pol'][i]
        
    invnorm = 1.0/lib.linalg.norm([xp,yp,zp], axis=0)
    
    point3D[..., 0] = invnorm*xp
    point3D[..., 1] = invnorm*yp
    point3D[..., 2] = invnorm*zp
    
    return point3D
        
def world2cam(point3D, o=None):
    if o is None:
        o = get_ocam_model()
    # This function assumes 1024 by 1024 sized images
    lib = torch if isinstance(point3D, torch.Tensor) else np
    
    # Pre-process UnrealEgo 3D coordinates
    if o["name"] == "unreal_ego_pose":
        point3D = UEp2CVp(point3D)
    
    shape = list(point3D.shape)
    shape[-1] = 2
    point2D = lib.zeros(tuple(shape))
    
    if isinstance(point3D, torch.Tensor):
        point2D = point2D.to(point3D.device)
    
    norm = lib.linalg.norm(point3D[..., :2], axis=-1)
    zeros_like_norm = lib.zeros_like(norm)
    n_zero = lib.isclose(norm, zeros_like_norm)
    n_nonzero = lib.logical_not(lib.isclose(norm, zeros_like_norm))
    
    # Handle normal cases
    theta = lib.arctan(point3D[n_nonzero][..., 2]/norm[n_nonzero])
    invnorm = 1.0/norm[n_nonzero]
    t = theta
    rho = lib.full(t.shape, o['invpol'][0])
    if isinstance(point3D, torch.Tensor):
        rho = rho.to(point3D.device)
    t_i = lib.ones_like(t)
    
    for i in range(1,o['length_invpol']):
        t_i *= t
        rho += t_i*o['invpol'][i]
        
    x = point3D[n_nonzero][..., 0]*invnorm*rho
    y = point3D[n_nonzero][..., 1]*invnorm*rho
    
    if isinstance(point3D, torch.Tensor):
        xy = lib.stack((x*o['c']+y*o['d']+o['xc'], x*o['e']+y+o['yc']), dim=-1)
    else:
        xy = lib.stack((x*o['c']+y*o['d']+o['xc'], x*o['e']+y+o['yc']), axis=-1)
    point2D[n_nonzero] = xy
    
    # Handle near zero cases
    zero_idx_1s = lib.ones_like(norm[n_zero])
    if isinstance(point3D, torch.Tensor):
        zero_xy = lib.stack((zero_idx_1s * o['xc'], zero_idx_1s * o['yc']), dim=-1)
    else:
        zero_xy = lib.stack((zero_idx_1s * o['xc'], zero_idx_1s * o['yc']), axis=-1)
    point2D[n_zero] = zero_xy
    
    if o["name"] == "unreal_ego_pose":
        point2D[..., 1] = o['yc'] * 2 - point2D[..., 1]
    
    return point2D

import cv2
limb_mask_indices_ue = [[2,4,6],
                     [3,5,7],
                     [8,10,12],
                     [9,11,13]]

limb_mask_indices_egocap = [[2,3,4],
                            [6,7,8],
                            [10,11,12],
                            [14,15,16]]


def get_limb_mask_indices(joint_preset):
    if joint_preset == "UnrealEgo":
        return limb_mask_indices_ue
    if joint_preset == "EgoCap":
        return limb_mask_indices_egocap


def generate_pseudo_limb_mask(pts2d, res=256, joint_preset=None):
    # Implement Pseudo-limb mask from EgoGlass Paper:
    '''
    We connect the areas between joints of Shoulder, Elbow, and Wrist to generate the mask for one
    arm and the areas between the joints of Hip, Knee and Ankle to generate the mask for one leg.
    '''
    
    thickness = 10
    # EgoCap: 10/15
    # UnrealEgo: 25/30
    
    thickness = thickness * res // 256
    limb_mask_indices = get_limb_mask_indices(joint_preset)
    mask = np.zeros((len(limb_mask_indices), res, res))
    pose = pts2d * res / 1024
    
    for i, limb in enumerate(limb_mask_indices):
        for parent, child in zip(limb[:-1], limb[1:]):
            parent_pose = tuple(map(int, pose[parent]))
            child_pose = tuple(map(int, pose[child]))
            color = 255  # White color in grayscale
            cv2.line(mask[i], tuple(parent_pose), tuple(child_pose), color, thickness)

    # Convert to binary mask
    binary_mask = (mask > 0).astype(np.float32)

    return binary_mask


def pose_to_2d_image(camera_pose, res=64, weight_depth=True, opt=None, side='left'):
    lib = torch if isinstance(camera_pose, torch.Tensor) else np
    
    # Project camera_pose to 2D.
    ocam = get_ocam_model(opt, side=side)
    camera_pose_2d = world2cam(camera_pose, o=ocam)
    camera_pose_depth = camera_pose[..., 2]
    
    # Draw lines for joints
    pose_image = lib.zeros((res, res), dtype=camera_pose.dtype)
    kinematic_parents = get_kinematic_parents(opt.joint_preset)
    for i in range(1, len(kinematic_parents)):
        line_image = lib.zeros_like(pose_image)
        parent_id = kinematic_parents[i]
        
        p_coord = lib.rint(camera_pose_2d[parent_id] * res / 1024).astype(lib.int32)
        coord = lib.rint(camera_pose_2d[i] * res / 1024).astype(lib.int32)
        p_depth = max(0, camera_pose_depth[parent_id])
        depth = max(0, camera_pose_depth[i])
        
        rr, cc, val = line_aa(p_coord[0], p_coord[1], coord[0], coord[1])
        
        # Apply depth weighting
        if weight_depth:
            p_distance = np.sqrt(np.square(rr - p_coord[0]) + np.square(cc - p_coord[1]))
            distance = np.sqrt(np.square(rr - coord[0]) + np.square(cc - coord[1]))
            if np.any(p_distance + distance == 0):
                t = 0
            else:
                t = p_distance / (p_distance + distance)
            val = val * ((1 - t) * p_depth + t * depth)
        
        valid_idx = np.logical_and(np.logical_and(rr >= 0, rr < res), np.logical_and(cc >= 0, cc < res))
        rr, cc, val = rr[valid_idx], cc[valid_idx], val[valid_idx]
        line_image[cc, rr] = val
        
        pose_image = np.maximum(pose_image, line_image)
    
    # Gaussian blur would be better for continuity.
    pose_image = gaussian_filter(pose_image, sigma=1)
    
    pose_image /= 0.15915589174187972
    
    return pose_image * 0.01


def heatmap_to_camera2d(heatmap):
    coord_shape = list(heatmap.shape)
    coord_shape.pop(-1)
    coord_shape[-1] = 2
    
    oos_idx = np.all(heatmap != 1.0, axis=(-1, -2))
    coords = np.zeros(coord_shape, dtype=np.float32)
    coords[oos_idx] = -1.0
    
    peak_pos = np.argwhere(heatmap == 1.0)
    for v in peak_pos:
        coords[tuple(v[:-2])] = v[-2:]
        
    return coords
    
                        
def UEp2CVp(coord):
    if isinstance(coord, torch.Tensor):
        coord = coord.clone()
    else: coord = coord.copy()
    coord[..., 1:] *= -1.0
    return coord

def coord2d_to_heatmap(coord2d, res=64, sigma=1.0):
    hm = np.zeros((coord2d.shape[0], res, res), dtype=np.float32)
    gaussian_margin = int(4 * sigma)
    margin_res = res + gaussian_margin * 2
    for i in range(coord2d.shape[0]):
        pos = coord2d[i] / 1024.0 * res
        x, y = pos[0], pos[1]
        
        expanded_hm = np.zeros((margin_res, margin_res), dtype=np.float32)
        if -4 <= y < res + 4 and -4 <= x < res:
            expanded_hm[int(y) + gaussian_margin, int(x) + gaussian_margin] = 1.0

        expanded_hm = gaussian_filter(expanded_hm, sigma=sigma)
        hm[i] = expanded_hm[gaussian_margin:-gaussian_margin, gaussian_margin:-gaussian_margin]
        
    hm /= 0.15915589174187972
    return hm
    
def get_pose_heatmap(camera_pose, res=64, o=None):
    hm = np.zeros((res, res), dtype=np.float32)
    poses = world2cam(camera_pose, o=o) * res / 1024
    for pos in poses:
        if 0 <= int(pos[1]) < res and 0 <= int(pos[0]) < res:
            hm[int(pos[1]), int(pos[0])] = 1.0
    hm = gaussian_filter(hm, sigma=1)
    hm /= 0.15915589174187972
    return hm

def sample_limb_heatmaps(camera_pose, res=64, weight_depth=False, depth_scale=1.0, depth_offset=0.0, opt=None, o=None, side='left'):
    # Try with two channel rot mask & slope
    kinematic_parents = get_kinematic_parents(opt.joint_preset)
    num_limbs = len(kinematic_parents)
    limb_heatmaps = np.zeros((num_limbs, res, res), dtype=np.float32)
    o = get_ocam_model(opt, side=side)
    camera_2d_pose = world2cam(camera_pose, o=o)
    camera_pose_depth = camera_pose[..., 2]
    
    for joint_idx in range(2,num_limbs+2):
        assign_idx = joint_idx - 2
        parent_idx = kinematic_parents[joint_idx]
        
        divider = (1024.0 / res)
        p_coord = camera_2d_pose[parent_idx]
        coord = camera_2d_pose[joint_idx]
        p_coord = np.rint(p_coord/divider).astype(int)
        coord = np.rint(coord/divider).astype(int)
        
        limb_heatmap = np.zeros((res, res), dtype=np.float32)
        
        rr, cc, val = line_aa(p_coord[0], p_coord[1], coord[0], coord[1])
        
        # Apply depth weighting
        if weight_depth:
            p_depth = max(0.0, camera_pose_depth[parent_idx])
            depth = max(0.0, camera_pose_depth[joint_idx])
            p_distance = np.sqrt(np.square(rr - p_coord[0]) + np.square(cc - p_coord[1]))
            distance = np.sqrt(np.square(rr - coord[0]) + np.square(cc - coord[1]))
            if np.any(np.isclose(p_distance + distance, 0.0)):
                val = val * np.minimum(p_depth, depth)
            else:
                t = p_distance / (p_distance + distance)
                val = val * ((1 - t) * p_depth + t * depth)
                val *= depth_scale
                val += depth_offset
        
        idx = np.logical_and(np.logical_and(rr >= 0, rr <= res-1), np.logical_and(cc >= 0, cc <= res-1))
        limb_heatmap[cc[idx], rr[idx]] = val[idx]
        limb_heatmap = gaussian_filter(limb_heatmap, sigma=1)

        limb_heatmaps[assign_idx] = limb_heatmap
    
    # Weight is not applied in this
    return limb_heatmaps
