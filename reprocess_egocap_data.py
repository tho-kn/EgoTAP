#!/usr/bin/env python
# coding: utf-8
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
import torch
from options.dataset_options import DatasetOptions
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
import json
from utils.util import normalize_input_img, get_index_to_joint_name
import h5py

# Create 3D Pose from calibration
def parse_egocap_calib(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        
    data = {}
    data['name'] = "egocap_pose"
    data['polynomialC2W'] = list(map(float, lines[2].strip().split()[1:]))
    data['polynomialW2C'] = list(map(float, lines[6].strip().split()[1:]))
    data['image_center'] = list(map(float, lines[10].strip().split()))
    data['affine'] = list(map(float, lines[14].strip().split()))
    data['size'] = list(map(int, lines[18].strip().split()))
    data['imageCircleRadius'] = 512
    
    return data


def get_calibration_data(data_dir, do_crop=False):
    calib_path0 = os.path.join(data_dir, 'cameraCalibration', 'stereo_c00_f_calibration.txt')
    calib_path1 = os.path.join(data_dir, 'cameraCalibration', 'stereo_c01_f_calibration.txt')
    calib0 = parse_egocap_calib(calib_path0)
    calib1 = parse_egocap_calib(calib_path1)
    
    if do_crop:
        # Fix image_center accounting for cropping
        calib0['orig_image_center'] = calib0['image_center'].copy()
        calib1['orig_image_center'] = calib1['image_center'].copy()

        h_center = int(calib0['image_center'][1] / 2) * 2
        calib0['image_center'][1] = calib0['image_center'][1] - h_center + 512
        h_center = int(calib1['image_center'][1] / 2) * 2
        calib1['image_center'][1] = calib1['image_center'][1] - h_center + 512
    
    return calib0, calib1


def get_cam1_extrinsics():
    # From cameraCalibration/calibrationFileV5.calibration
    s = "-6.811572770603570E-01 -1.978935067001849E-01 -7.048850430485982E-01 -2.369619435605097E+01  -1.823782681785945E-01 -8.865639851773102E-01 4.251381745770462E-01 4.735684810307217E+02  -7.090577770562122E-01 4.181416747855088E-01 5.677980350473865E-01 1.699215408061096E+01  0 0 0 1"
    # split the string into a list of strings, convert to floats and reshape to 4x4
    matrix = np.array(s.split(), dtype=float).reshape(4, 4)
    
    def pre_cond_RT(RT):
        # From EgoCapDataloader
        coordinateTransformation4 = np.eye(4)
        coordinateTransformation4[2,2] = -1; # because of negative z direction convention

        # World coordinate system to camera coordinate system
        RT = coordinateTransformation4 @ RT @ coordinateTransformation4
        return RT
    
    matrix = pre_cond_RT(matrix)

    print("Cam1 Extrinsic: ", matrix)
    return matrix


def crop_resize_images(ocam, images, do_crop=False):
    # crop images to 512 by 512 from the center
    # calibration file says images are 1280 by 1024 but it is 640 by 512
    
    if do_crop:
        if images.shape[2] == 512:
            h_center = int(ocam['image_center'][1] / 2)
            images = images[:, :, :, h_center - 256: h_center + 256]
            
        elif images.shape[2] == 1024:
            h_center = int(ocam['image_center'][1])
            images = images[:, :, :, h_center - 512: h_center + 512]
    
    # Images: (Batch, 3, 512, 512)
    images = torch.nn.functional.interpolate(torch.tensor(images), size=(256, 256), mode='bilinear', align_corners=False)
    
    return images.numpy()

        
from utils.projection import world2cam, cam2world, coord2d_to_heatmap, get_ocam_model
from utils.data import get_local_rot, overwrite_limb_data
import tqdm


# Function to process image
def process_img(img_path, ocam, do_crop=False, flip=False):
    img = Image.open(img_path)
    img = np.array(img)
    if flip:
        img = np.flip(img, axis=1).copy()
    img = img.transpose((2, 0, 1)) # Convert to (Channel, Height, Width)
    img = np.expand_dims(img, axis=0) # Add batch dimension
    img = crop_resize_images(ocam, img, do_crop)
    img = img.squeeze(0)
    return img


# Function to process coordinates
# Recenter coordinate to cropped image
def process_coordinates(coordinates, ocam, do_crop=False):
    coordinates = coordinates.copy()
    if do_crop:
        h_center = int(ocam['image_center'][1] / 2) * 2
        coordinates[..., 0] = coordinates[..., 0] - h_center + 512
        
    # Return coordinates in 1024 by 1024
    return coordinates


def parse_2d_datafile(annotation_file):
    # Read the annotation file
    with open(annotation_file, 'r') as f:
        lines = f.readlines()

    # Process the annotation file
    data = []
    for line in lines:
        line = line.strip()
        if line.startswith('#'):
            data.append({})
        elif './images/' in line:
            data[-1]['img_path'] = line.strip()
        elif len(line) == 0:
            continue
        elif line.isdigit():
            key = 'num_coordinates' if (
                'dimensions' in data[-1] and isinstance(data[-1]['dimensions'], list) and len(data[-1]['dimensions']) == 3) \
                else 'dimensions'
            if key not in data[-1]:
                data[-1][key] = int(line.strip())
            else:
                val = data[-1][key]
                if isinstance(val, int):
                    data[-1][key] = [val, int(line.strip())]
                else:
                    data[-1][key] = val + [int(line.strip())]
        else:
            if 'coordinates' not in data[-1]:
                data[-1]['coordinates'] = []
            coord = tuple(map(float, line.split()))
            
            # Calibration is done in full res so multiply 2
            x = coord[2] * 2.0
            y = coord[1] * 2.0
        
            # Convert to (x, y) format
            # Somehow this order seems correct, x/y axis might not match from intuition
            data[-1]['coordinates'].append((y, x))
        
    for i in range(len(data)):
        data[i]['coordinates'] = np.array(data[i]['coordinates'], dtype=np.float32)
    
    return data
    
joint_reorder = [0, 1, 6, 7, 8, 9, 2, 3, 4, 5, 14, 15, 16, 17, 10, 11, 12, 13]
    
    
def flip_raw_coordinates(coords):
    # The annotation is integer annotation per pixel
    # Maximum value is width - 1
    coords = coords.copy()
    coords[..., 0] = 1280.0 - coords[..., 0]
    coords = coords[joint_reorder]
    return coords


import copy
def get_orig_ocam(ocam):
    orig_ocam = copy.deepcopy(ocam)
    orig_ocam["image_center"] = orig_ocam["orig_image_center"]
    return orig_ocam


ds_path = "./EgoCapDataloader3D/Ego_pose_stereo_cleaned.hdf5"
hf = h5py.File(ds_path, 'r')

def gen_im_name(S, cam, frame, aug=False):
    if aug:
        v = 2
    else:
        v = 0
    im_name = "images/S%d_v00%d_cam%d_frame-%04d.jpg" % (S,v,cam,frame)
    return im_name


def process_dataset(opt, dataset_dir, ocam0, ocam1, cam1E):
    orig_ocam0 = get_orig_ocam(ocam0)
    orig_ocam1 = get_orig_ocam(ocam1)
    
    data_size = range(hf['pose_2d'][...].shape[0])
    if opt.experiment:
        data_size = range(10)
    data_idx = tqdm.tqdm(data_size)
    
    # Ensure output directory exists
    output_dir = dataset_dir
    os.makedirs(output_dir, exist_ok=True)
    
    for idx in data_idx:
        subject_id = hf['subject_index'][idx]
        frame_id = [hf['frame_index'][idx,0], hf['frame_index'][idx,1]]
        img_paths = [gen_im_name(subject_id,i,frame_id[i]) for i in range(2)]
        data_idx.set_description(f"Processing {img_paths[0]} and {img_paths[1]}")
        
        npy_path = os.path.join(output_dir, f'S{subject_id}', opt.data_sub_path, f'frame_{frame_id[0]}.npy')
        os.makedirs(os.path.dirname(npy_path), exist_ok=True)
        
        # Cam1 Image is flipped, flip it back
        orig_img_0 = process_img(os.path.join(dataset_dir, img_paths[0]), orig_ocam0, do_crop=opt.do_crop, flip=False)
        orig_img_1 = process_img(os.path.join(dataset_dir, img_paths[1]), orig_ocam1, do_crop=opt.do_crop, flip=True)
        img_0 = normalize_input_img(orig_img_0)
        img_1 = normalize_input_img(orig_img_1)

        # 2D Coordinates in full resolution
        raw_coords_0 = np.array(hf['pose_2d'][idx][0] * [1280, 1024])
        raw_coords_1 = np.array(hf['pose_2d'][idx][1] * [1280, 1024])
        
        # The 2D poses are for pixels
        coordinates_0 = process_coordinates(raw_coords_0, orig_ocam0, do_crop=opt.do_crop)
        coordinates_1 = process_coordinates(raw_coords_1, orig_ocam1, do_crop=opt.do_crop)
        
        # Get Heatmaps
        heatmap_left = coord2d_to_heatmap(coordinates_0[1:], res=64)
        heatmap_right = coord2d_to_heatmap(coordinates_1[1:], res=64)
        
        local_pose = np.array(hf['pose_3d'][idx]) / 10.0
        
        local_pose1 = np.concatenate([local_pose, np.ones((local_pose.shape[0], 1))], axis=-1) @ cam1E
        local_pose[..., 2] *= -1.0
        local_pose1[..., 2] *= -1.0
        
        local_rot = get_local_rot(opt, local_pose)
        
        pelvis_left = np.zeros((3))
        pelvis_right = np.zeros((3)) # Cam0 center in Cam1 coords

        # Wrap in a dictionary and save as numpy array
        data_dict = {'input_rgb_left': img_0,
                     'input_rgb_right': img_1,
                     'gt_heatmap_left': heatmap_left,
                     'gt_heatmap_right': heatmap_right,
                     'gt_camera_2d_left': coordinates_0,
                     'gt_camera_2d_right': coordinates_1,
                     'gt_local_rot': local_rot,
                     'gt_local_pose': local_pose,
                     'gt_global_pose': local_pose, # We don't have GT global pose
                     'gt_pelvis_left': pelvis_left,
                     'gt_pelvis_right': pelvis_right,
        }
        
        # pts2d_left, pts2d_right, pts3d_left, pts3d_right = coordinates_0, coordinates_1, local_pose, local_pose1
        pts2d_left, pts2d_right, pts3d_left, pts3d_right = coordinates_0, coordinates_1, local_pose, local_pose
        
        overwrite_limb_data(data_dict, pts2d_left, pts2d_right, pts3d_left, pts3d_right,
                           htype='line', sigma=1, joint_preset=opt.joint_preset)
        
        if not opt.experiment:
            np.save(npy_path, np.array(data_dict))
        
    print("Showing samples from train set")
    
    itjd = get_index_to_joint_name("EgoCap")
    itj = [None] * len(itjd)
    for k, v in itjd.items():
        itj[k] = v


def process_validate_dataset(opt, dataset_dir, dataset_dir_2d, ocam0, ocam1, cam1E):
    orig_ocam0 = get_orig_ocam(ocam0)
    orig_ocam1 = get_orig_ocam(ocam1)
    cam1E_inv = np.linalg.inv(cam1E)
    
    # Ensure output directory exists
    output_dir = os.path.join(dataset_dir, opt.data_sub_path)
    os.makedirs(output_dir, exist_ok=True)
    
    annotation_file_2D = os.path.join(dataset_dir_2d, "dataset.txt")
    data_2D = parse_2d_datafile(annotation_file_2D)
    
    data_2D_dict = {}
    data_pixel_dict = {}
    
    # Make a dict that maps image to coordinates
    # Flip coordinates horizontally
    
    for i, val in enumerate(data_2D):
        if "S7" in val["img_path"]:
            coords = val["coordinates"]
            if "cam1" in val["img_path"]:
                coords = flip_raw_coordinates(coords)
            data_2D_dict[val["img_path"]] = coords
            
            ocam = orig_ocam0 if 'cam0' in val["img_path"] else orig_ocam1
            camera_2d = process_coordinates(coords, ocam, do_crop=opt.do_crop)
            data_pixel_dict[val["img_path"]] = camera_2d
    
    annotation_file = os.path.join(dataset_dir, "dataset3D.mddd")
    with open(annotation_file, 'r') as f:
        lines = f.readlines()
    
    data = []
    for line in lines[1:]:
        line = line.strip()
        if line.startswith('Skeletool'):
            continue
        coord = tuple(map(float, line.split()))
        if len(coord) <= 1:
            continue
        if len(coord) != 55:
            print("Error: ", line, len(coord))
            continue
        else:
            frame = int(coord[0])
            data.append({
                'frame': frame,
                'frame1': frame - 85,
                'img_path0': os.path.join(dataset_dir, 'images', 'franzi_01-cam0', f"frame-{frame}.jpg"),
                'img_path1': os.path.join(dataset_dir, 'images', 'franzi_01-cam1', f"frame-{frame - 85}.jpg"),
                'gt_local_pose': np.array(coord[1:], dtype=np.float32).reshape(-1, 3)
            })
            
    if opt.experiment:
        data = data[-24:]

    # Process pairs of images
    data_idx = tqdm.tqdm(range(0, len(data)))
    for i in data_idx:
        data_idx.set_description(f"Processing {data[i]['img_path0']} and {data[i]['img_path1']}")
        npy_path = os.path.join(output_dir, f"frame_{data[i]['frame']}.npy")
        
        orig_img_0 = process_img(os.path.join(dataset_dir, data[i]['img_path0']), orig_ocam0, do_crop=opt.do_crop)
        orig_img_1 = process_img(os.path.join(dataset_dir, data[i]['img_path1']), orig_ocam1, do_crop=opt.do_crop)
        img_0 = normalize_input_img(orig_img_0)
        img_1 = normalize_input_img(orig_img_1)
        
        data_2d_path_0 = "./images/S7_v003_cam0_frame-{}.jpg".format(data[i]['frame'])
        data_2d_path_1 = "./images/S7_v003_cam1_frame-{}.jpg".format(data[i]['frame1'])
        
        # 2D Coordinates in full resolution, already flipped
        raw_coords_0 = data_2D_dict[data_2d_path_0]
        raw_coords_1 = data_2D_dict[data_2d_path_1]
        
        # The 2D poses are for pixels
        coordinates_0 = process_coordinates(raw_coords_0, orig_ocam0, do_crop=opt.do_crop)
        coordinates_1 = process_coordinates(raw_coords_1, orig_ocam1, do_crop=opt.do_crop)
        
        # Get Heatmaps
        heatmap_left = coord2d_to_heatmap(coordinates_0[1:], res=64)
        heatmap_right = coord2d_to_heatmap(coordinates_1[1:], res=64)
        
        local_pose = data[i]['gt_local_pose']/ 10.0 # Validation and Train 3D Z axis is flipped
        local_rot = get_local_rot(opt, local_pose)
        
        local_pose_norm = local_pose / np.linalg.norm(local_pose, axis=-1)[..., None]
        
        pelvis_left = np.zeros((3))
        pelvis_right = np.zeros((3)) # Ignore cam1 transform

        # Wrap in a dictionary and save as numpy array
        data_dict = {'input_rgb_left': img_0,
                     'input_rgb_right': img_1,
                     'gt_heatmap_left': heatmap_left,
                     'gt_heatmap_right': heatmap_right,
                     'gt_camera_2d_left': coordinates_0, # This contains cropped coord 1024
                     'gt_camera_2d_right': coordinates_1,
                     'gt_local_rot': local_rot,
                     'gt_local_pose': local_pose, # Fix unit to cm like UnrealEgo
                     'gt_global_pose': local_pose, # We don't have GT global pose
                     'gt_pelvis_left': pelvis_left,
                     'gt_pelvis_right': pelvis_right,
        }
        
        local_pose1 = local_pose
        pts2d_left, pts2d_right, pts3d_left, pts3d_right = coordinates_0, coordinates_1, local_pose, local_pose1
        overwrite_limb_data(data_dict, pts2d_left, pts2d_right, pts3d_left, pts3d_right,
                           htype='line', sigma=1, joint_preset=opt.joint_preset)
    
        if not opt.experiment:
            np.save(npy_path, np.array(data_dict))
        
    print("Showing Samples from test set")
    
    itjd = get_index_to_joint_name("EgoCap")
    itj = [None] * len(itjd)
    for k, v in itjd.items():
        itj[k] = v


def modify_dataset(opt):
    training_set_dir = os.path.join(opt.data_dir, "training_v000")
                        # os.path.join(opt.data_dir, "training_v002")
    validation_set_2d_dir = os.path.join(opt.data_dir, "validation_v003_2D")
    validation_set_3d_dir = os.path.join(opt.data_dir, "validation_v003_3D")
    
    calib_data0, calib_data1 = get_calibration_data(validation_set_3d_dir, do_crop=opt.do_crop)
    
    cam1E = get_cam1_extrinsics()

    if opt.do_crop:
        orig_image_center0 = calib_data0['orig_image_center']
        orig_image_center1 = calib_data1['orig_image_center']
        calib_data0.pop('orig_image_center')
        calib_data1.pop('orig_image_center')
    calib_json_path0 = os.path.join(opt.data_dir, "fisheye.calibration_left.json")
    calib_json_path1 = os.path.join(opt.data_dir, "fisheye.calibration_right.json")
    
    with open(calib_json_path0, "w") as f:
        json.dump(calib_data0, f)
    with open(calib_json_path1, "w") as f:
        json.dump(calib_data1, f)
    
    calib_data0 = get_ocam_model(opt, 'left')
    calib_data1 = get_ocam_model(opt, 'right')
    
    if opt.do_crop:
        calib_data0['orig_image_center'] = orig_image_center0
        calib_data1['orig_image_center'] = orig_image_center1
        
    process_validate_dataset(opt, validation_set_3d_dir, validation_set_2d_dir, calib_data0, calib_data1, cam1E)
    print("Finished processing validation set!")
    
    # Train set has 640 x 512 resolution
    # Crop those horizontally, and rescale them to 256 x 256
    process_dataset(opt, training_set_dir, calib_data0, calib_data1, cam1E)
    print("Finished processing training set!")

    train_list_txt = os.path.join(opt.data_dir, "train.txt")
    
    from pathlib import Path
    with open(train_list_txt, "w") as f:
        for i in range(1, 7):
            f.write(str(Path(opt.data_dir) / "training_v000" / f"S{i}"))
            f.write("\n")
    
    val_list_txt = os.path.join(opt.data_dir, "validation.txt")
    with open(val_list_txt, "w") as f:
        f.write(str(Path(opt.data_dir) / "validation_v003_3D"))
        
    test_list_txt = os.path.join(opt.data_dir, "test.txt")
    with open(test_list_txt, "w") as f:
        f.write(str(Path(opt.data_dir) / "validation_v003_3D"))


from options.dataset_options import DatasetOptions

if __name__ == "__main__":
    opt = DatasetOptions().parse()
    opt.do_crop = True
    modify_dataset(opt)

