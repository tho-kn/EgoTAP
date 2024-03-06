import os
import numpy as np
from scipy.ndimage import gaussian_filter
from utils.projection import world2cam
from utils.util import get_index_to_joint_name, get_kinematic_parents

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def process_npy_path(opt, npy_path):
    npy_subpath = npy_path.replace(opt.data_dir, "", 1).replace(opt.data_sub_path, "", 1).replace(".npy", "", 1)
    npy_name = npy_subpath.replace("/", "-").replace("\\", "-").replace(".", "-")

    [head, tail] = os.path.split(npy_path)
    take_data_dir = os.path.join(head, os.pardir)

    json_path = os.path.join(os.path.join(take_data_dir, 'json'), tail[:-4] + ".json")

    return npy_subpath, npy_name, head, tail, take_data_dir, json_path

def get_num_joints(opt):
    joint_preset = opt.joint_preset
    index_to_joint_name = get_index_to_joint_name(joint_preset)
    num_joints = len(index_to_joint_name)
    return num_joints

def get_local_rot(opt, pose3d):
    num_joints = get_num_joints(opt)
    joint_orient = np.zeros(shape=(num_joints, 3), dtype=np.float32)
    for i in range(1, num_joints):
        joint_pos_delta = np.array(pose3d[i]) - np.array(pose3d[get_kinematic_parents(opt.joint_preset)[i]])
        joint_orient[i] = joint_pos_delta / np.linalg.norm(joint_pos_delta, axis=-1)
    return joint_orient

def vec2vecR(v1, v2):
    """
    Returns the rotation matrix that rotates v1 to v2
    Two vectors must be of the same length
    """
    u1 = v1/np.linalg.norm(v1)
    u2 = v2/np.linalg.norm(v2)
    v = np.cross(u1, u2)
    s = np.linalg.norm(v)
    c = np.dot(u1, u2)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    r = np.eye(3) + vx + np.dot(vx, vx) * (1 - c) / (s ** 2)
    return r

def UR2R(theta):
    # Convert Unreal Engine Rotator to Rotation Matrix
    rad_angle = np.deg2rad(theta)
    sp, cp = np.sin(rad_angle[..., 0]), np.cos(rad_angle[..., 0])
    sy, cy = np.sin(rad_angle[..., 1]), np.cos(rad_angle[..., 1])
    sr, cr = np.sin(rad_angle[..., 2]), np.cos(rad_angle[..., 2])

    if len(rad_angle.shape) == 1:
        R = np.zeros(shape=(3,3,), dtype=np.float32)
    else:
        shape = rad_angle.shape[:-1]
        R_shape = list(shape)
        R_shape.extend([3, 3])
        R = np.zeros(shape=tuple(R_shape), dtype=np.float32)

    R[..., 0, 0] = cp * cy
    R[..., 0, 1] = cp * sy
    R[..., 0, 2] = sp
    R[..., 1, 0] = sr * sp * cy - cr * sy
    R[..., 1, 1] = sr * sp * sy + cr * cy
    R[..., 1, 2] = -sr * cp
    R[..., 2, 0] = -(cr * sp * cy + sr * sy)
    R[..., 2, 1] = cy * sr - cr * sp * sy
    R[..., 2, 2] = cr * cp

    return R

def pts2d_to_heatmap(coord, res=64, area=64):
    heatmap = np.zeros((area, area), dtype=np.float32)
    if len(coord.shape) == 1:
        coord = coord[None, :]
    
    coords = coord
    
    for coord in coords:
        hm_coord = np.rint(coord/(1024.0 / res)).astype(int)
        padding = (area - res) // 2
        hm_coord += padding

        if 0 <= hm_coord[0] < area and 0 <= hm_coord[1] < area:
            heatmap[hm_coord[1], hm_coord[0]] = 1
            heatmap = gaussian_filter(heatmap, sigma=1)
            heatmap /= 0.15915589174187972
    
    return heatmap

from utils.projection import get_ocam_model
def pts3d_to_heatmap(coord, res=64, area=64, opt=None):
    ocam = get_ocam_model(opt)
    coord_2d = world2cam(coord, ocam)
    heatmap = pts2d_to_heatmap(coord_2d, res, area)
    
    return heatmap

def add_pelvis_heatmap(npy_item, joint_data):
    num_heatmaps_l = npy_item['gt_heatmap_left'].shape[0]
    num_heatmaps_r = npy_item['gt_heatmap_right'].shape[0]
    gt_heatmap_left = np.ndarray(shape=(num_heatmaps_l + 1,64,64), dtype=np.float32)
    gt_heatmap_right = np.ndarray(shape=(num_heatmaps_r + 1,64,64), dtype=np.float32)
    
    gt_heatmap_left[:num_heatmaps_l] = npy_item['gt_heatmap_left']
    gt_heatmap_right[:num_heatmaps_r] = npy_item['gt_heatmap_right']
    
    pelvis_2d_coord_left = np.array(joint_data['pelvis']['camera_left_pts2d'])
    pelvis_2d_coord_right = np.array(joint_data['pelvis']['camera_right_pts2d'])
    
    gt_heatmap_left[num_heatmaps_l] = pts2d_to_heatmap(pelvis_2d_coord_left)
    gt_heatmap_right[num_heatmaps_r] = pts2d_to_heatmap(pelvis_2d_coord_right)
    
    return gt_heatmap_left, gt_heatmap_right


def _fpart(x):
    return x - int(x)

def _rfpart(x):
    return 1 - _fpart(x)
    
def draw_line_aa(p1, p2):
    """Draws an anti-aliased line in img from p1 to p2."""
    x1, y1 = p1
    x2, y2 = p2
    dx, dy = x2-x1, y2-y1
    steep = abs(dx) < abs(dy)
    p = lambda px, py: ((px,py), (py,px))[steep]

    if steep:
        x1, y1, x2, y2, dx, dy = y1, x1, y2, x2, dy, dx
    if x2 < x1:
        x1, x2, y1, y2 = x2, x1, y2, y1

    grad = dy/dx
    intery = y1 + _rfpart(x1) * grad
    def draw_endpoint(pt):
        x, y = pt
        xend = round(x)
        yend = y + grad * (xend - x)
        xgap = _rfpart(x + 0.5)
        px, py = int(xend), int(yend)
        pixs = [p(px, py), p(px, py+1)], [_rfpart(yend) * xgap, _fpart(yend) * xgap]
        return px, pixs
    
    coords, val = [], []

    xstart, spixs = draw_endpoint(p(*p1))
    xstart += 1
    coords.extend(spixs[0])
    val.extend(spixs[1])
    xend, epixs = draw_endpoint(p(*p2))
    coords.extend(epixs[0])
    val.extend(epixs[1])

    for x in range(xstart, xend):
        y = int(intery)
        coords.append(p(x, y)), coords.append(p(x, y+1))
        val.append(_rfpart(intery)), val.append(_fpart(intery))
        intery += grad
        
    row = np.array([c[0] for c in coords], dtype=np.int32)
    col = np.array([c[1] for c in coords], dtype=np.int32)
    val = np.array(val, dtype=np.float32)
        
    return row, col, val


from skimage.draw import line_aa
def get_line_limb_heatmap(p_coord, coord, limb_heatmap=None, res=64):
    if limb_heatmap is None:
        limb_heatmap = np.zeros((res, res))
    p_coord = np.rint(p_coord).astype(int)
    coord = np.rint(coord).astype(int)
    rr, cc, val = line_aa(p_coord[0], p_coord[1], coord[0], coord[1])
    
    idx = np.logical_and(np.logical_and(rr >= 0, rr <= res-1), np.logical_and(cc >= 0, cc <= res-1))
    limb_heatmap[cc[idx], rr[idx]] = val[idx]
    
    return limb_heatmap


def get_points_limb_heatmap(p_coord, coord, limb_heatmap=None, res=64, area=64):
    if limb_heatmap is None:
        limb_heatmap = np.zeros((area, area))
    heatmap = pts2d_to_heatmap(np.stack((p_coord, coord)), res, area)
    limb_heatmap += heatmap
        
    return limb_heatmap
        
        
def get_limb_data(pts2d, pts3d, res=64, area=None, htype='line', sigma=1, joint_preset="UnrealEgo", ocam=None):
    # Try with two channel rot mask & slope
    index_to_joint_name = get_index_to_joint_name(joint_preset)
    kinematic_parents = get_kinematic_parents(joint_preset)
    num_joints = len(index_to_joint_name)
    
    if area is None:
        area = res
    limb_heatmaps = np.zeros((num_joints - 1, area, area), dtype=np.float32)
    lengths = np.zeros(num_joints - 1, dtype=np.float32)
    theta = np.zeros(num_joints - 1, dtype=np.float32)
    
    if (area - res) % 2 != 0:
        print('area - res must be even number')
        exit()
    padding = (area - res) // 2
    
    for joint_idx in range(1, num_joints):
        assign_idx = joint_idx - 1
        parent_idx = kinematic_parents[joint_idx]
        
        divider = (1024.0 / res)
        p_coord = pts2d[parent_idx]
        coord = pts2d[joint_idx]
        p_coord = p_coord/divider
        coord = coord/divider
        p_coord3d = pts3d[parent_idx]
        coord3d = pts3d[joint_idx]
        
        # sign = p_coord3d[2] > coord3d[2]
        limb_length = np.linalg.norm(p_coord3d - coord3d)
        limb_3d = p_coord3d - coord3d
        limb_2dlen = np.linalg.norm(limb_3d[:2])
        theta[assign_idx] = np.arctan(limb_3d[2]/limb_2dlen)
        
        limb_heatmap = np.zeros((res, res), dtype=np.float32)
        limb_pixel_length = np.linalg.norm(p_coord - coord) + 1.0
    
        p_coord += padding
        coord += padding
        
        if htype == 'line':
            lengths[assign_idx] = limb_pixel_length
            limb_heatmap = get_line_limb_heatmap(p_coord, coord, limb_heatmap, res)
        elif htype == 'points':
            lengths[assign_idx] = 2
            limb_heatmap = get_points_limb_heatmap(p_coord, coord, limb_heatmap, res)
        else:
            raise Exception("Undefined limb heatmap type")
        
        limb_heatmap = gaussian_filter(limb_heatmap, sigma=sigma, mode='constant')
        limb_heatmap *= sigma
        
        limb_heatmaps[assign_idx] = limb_heatmap
    
    return limb_heatmaps, lengths, theta

def overwrite_limb_data(npy_item, pts2d_left, pts2d_right, pts3d_left, pts3d_right,
                        res=64, area=64, htype='line', sigma=1, joint_preset=None, is_stereo=True):
    (npy_item['gt_limb_heatmap_left'],
        npy_item['gt_pixel_length_left'],
        npy_item['gt_limb_theta']) = get_limb_data(pts2d_left, pts3d_left, res, area, htype, sigma=sigma, joint_preset=joint_preset)
    if is_stereo:
        (npy_item['gt_limb_heatmap_right'], \
            npy_item['gt_pixel_length_right'],
            _) = get_limb_data(pts2d_right, pts3d_right, res, area, htype, sigma=sigma, joint_preset=joint_preset)

import torch
def convert_norm_angle_to_rgb(cos_hm, sin_hm):
    norm_hm = torch.sqrt(cos_hm**2 + sin_hm**2)
    angle_hm = torch.atan2(sin_hm, cos_hm)
    
    rgb_norm_angle_hm = torch.zeros_like(cos_hm)
    rgb_norm_angle_hm = rgb_norm_angle_hm.unsqueeze(-1)
    # repeat last dimension to 3 without any assumption of shape
    rgb_norm_angle_hm = rgb_norm_angle_hm.expand(*rgb_norm_angle_hm.shape[:-1], 3)
    rgb_norm_angle_hm[..., 0] = angle_hm / np.pi
    rgb_norm_angle_hm[..., 1] = norm_hm * 2 - 1
    rgb_norm_angle_hm[..., 2] = 1.0
    return rgb_norm_angle_hm
