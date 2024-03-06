import os
import numpy as np
import torch
from options.dataset_options import DatasetOptions
from dataloader.image_folder import make_dataset
import operator
from tqdm import tqdm
from utils.util import RunningAverageStdDict, try_json
from utils.data import *
import shutil
import time, subprocess
from copy import deepcopy

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def modify_dataset(opt, id=None, overwrite=False):
    if not os.path.exists(opt.data_dir):
        os.makedirs(opt.data_dir)
        print(f"Folder {opt.data_dir} created!")
    else:
        print(f"Folder {opt.data_dir} already exists!")
        
    # load_size_rgb = opt.load_size_rgb
    # load_size_heatmap = opt.load_size_heatmap
    opt.joint_preset = "UnrealEgo"

    npy_data_sequences = []
    num_npy_sequences = 0
    fail_list_path = os.path.join(opt.data_dir, 'fails.txt' if not opt.experiment else 'exp_fails.txt')
    fail_file = open(fail_list_path, "w")

    for mode in ['train', 'test', 'validation']:
        data_list_path = os.path.join(opt.metadata_dir[0], mode + '.txt')  # CT\UnrealEgo\static00\UnrealEgoData\train.txt
        fail_list_path = fail_list_path.replace(opt.default_data_path, opt.data_dir, 1)
        
        npy_dir = opt.data_sub_path
        mode_npy_data_sequences, mode_num_npy_sequences = make_dataset(
            opt=opt, 
            data_list_path=data_list_path, 
            data_sub_path=npy_dir,
            id=id,
            as_sequence=True,
            use_metadata=True,
        )
        npy_data_sequences.extend(mode_npy_data_sequences)
        print("Found {} sequences in {} mode".format(mode_num_npy_sequences, mode))
        num_npy_sequences += mode_num_npy_sequences

    file_to_copy = ["train.txt", "test.txt", "validation.txt"]
    for file in file_to_copy:
        shutil.copyfile(os.path.join(opt.metadata_dir[0], file), os.path.join(opt.data_dir, file))
    shutil.copyfile(os.path.realpath(__file__), os.path.join(opt.data_dir, 'script.py'))
    
    with open(os.path.join(opt.data_dir, 'modify_dataset_log.txt'), 'w') as f:
        f.write(time.strftime('%l:%M%p %Z on %b %d, %Y\n'))
        branch_name = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], cwd=os.path.dirname(os.path.realpath(__file__)))
        f.write(branch_name.decode('ascii').strip() + "\n")
        hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=os.path.dirname(os.path.realpath(__file__)))
        f.write(hash.decode('ascii').strip() + "\n")

    fail_cnt = 0
    stat_dict = RunningAverageStdDict()

    for file_idx, seq_npy_paths in tqdm(enumerate(npy_data_sequences), total=len(npy_data_sequences), desc="Modifying dataset"):
        # First frame info is not needed since any part of sequence can be a first frame during training
        npy_paths = []
        npy_datas = []
        json_paths = []
        json_datas = []
        fail = False
        fail_json_path = ""

        for index in range(len(seq_npy_paths)):
            npy_path = seq_npy_paths[index]
            npy_data = np.load(npy_path, allow_pickle=True)
            json_path = process_npy_path(opt, npy_path)[5]
            json_data = try_json(json_path)

            if json_data == None:
                fail = True
                fail_json_path = json_path
                break
            
            if not fail:
                npy_paths.append(npy_path)
                npy_datas.append(npy_data)
                json_paths.append(json_path)
                json_datas.append(json_data)
        
        if fail: 
            fail_cnt += 1
            print(fail_json_path, ": Failed to find required json file")
            
            fail_file.write("{}\n".format(fail_json_path))
            continue

        _, _, head, _, _, _ = process_npy_path(opt, npy_paths[0])
        npy_new_dir = None
        for metadir in opt.metadata_dir:
            if metadir in head:
                npy_new_dir = head.replace(metadir, opt.data_dir)
                
        if npy_new_dir is None:
            print("Failed to find metadata directory in {}".format(head))
            continue

        os.makedirs(npy_new_dir, exist_ok=True)

        root_translations = []

        # Sequences are natsorted
        for index in range(len(seq_npy_paths)):
            _, npy_name, _, tail, _, json_path = process_npy_path(opt, npy_paths[index])
            npy_data = npy_datas[index]
            npy_item = deepcopy(npy_data.item())
            json_data = json_datas[index]

            # all_joint_names = curr_json_data['joints'].keys()
            ground_z_value = json_data['ground_z_value']
            joint_data = json_data['joints']

            root_translation = np.array(joint_data['pelvis']['trans'])
            root_translation[2] -= ground_z_value
            root_rotation = UR2R(np.array(joint_data['pelvis']['rot']))

            pelvis_camera_coord = list(map(operator.add, joint_data['pelvis']['camera_left_pts3d'], 
                                                joint_data['pelvis']['camera_right_pts3d']))
            pelvis_camera_coord = np.array(list(map(lambda x: x/2.0, pelvis_camera_coord)))
            
            npy_item['gt_heatmap_left'], npy_item['gt_heatmap_right'] = add_pelvis_heatmap(npy_item, joint_data)
            
            npy_item['gt_pelvis_left'] = np.array(joint_data['pelvis']['camera_left_pts3d'])
            npy_item['gt_pelvis_right'] = np.array(joint_data['pelvis']['camera_right_pts3d'])

            pts2d_left = np.ndarray(shape=(16,2), dtype=np.float32)
            pts3d_left = np.ndarray(shape=(16,3), dtype=np.float32)
            pts2d_right = np.ndarray(shape=(16,2), dtype=np.float32)
            pts3d_right = np.ndarray(shape=(16,3), dtype=np.float32)
            index_to_joint_name = get_index_to_joint_name(joint_preset=opt.joint_preset)
            
            for i in range(16):
                pts2d_left[i] = np.array(joint_data[index_to_joint_name[i]]['camera_left_pts2d'])
                pts3d_left[i] = np.array(joint_data[index_to_joint_name[i]]['camera_left_pts3d'])
                pts2d_right[i] = np.array(joint_data[index_to_joint_name[i]]['camera_right_pts2d'])
                pts3d_right[i] = np.array(joint_data[index_to_joint_name[i]]['camera_right_pts3d'])
    
            overwrite_limb_data(npy_item, pts2d_left, pts2d_right, pts3d_left, pts3d_right, htype='line', joint_preset="UnrealEgo")
            npy_item['raw_joint_data'] = joint_data

            # Global coordinate, root coordinate projected to XY plane, camera coordinate
            global_pose = np.ndarray(shape=(16,3), dtype=np.float32)
            gt_camera_2d_left = np.ndarray(shape=(16,2), dtype=np.float32)
            gt_camera_2d_right = np.ndarray(shape=(16,2), dtype=np.float32)
            
            # Get global pose of joints
            for i in range(16):
                joint_name = index_to_joint_name[i]
                joint_global_pose = np.array(joint_data[joint_name]['trans'])
                joint_global_pose[2] -= ground_z_value
                global_pose[i] = joint_global_pose
                
                joint_camera_2d_left = np.array(joint_data[joint_name]['camera_left_pts2d'])
                joint_camera_2d_right = np.array(joint_data[joint_name]['camera_right_pts2d'])
                gt_camera_2d_left[i] = joint_camera_2d_left
                gt_camera_2d_right[i] = joint_camera_2d_right

            npy_item['gt_global_pose'] = global_pose
            
            npy_item['gt_camera_2d_left'] = gt_camera_2d_left
            npy_item['gt_camera_2d_right'] = gt_camera_2d_right

            root_translations.append(root_translation)

            if index == 0:
                continue

            root_translation_delta = root_translations[-1] - root_translations[-2]

            npy_item['gt_root_translation'] = root_translation
            npy_item['gt_root_translation_delta'] = root_translation_delta
            npy_item['gt_root_rotation'] = root_rotation
            npy_item['name'] = npy_name
            
            npy_item['gt_local_rot'] = get_local_rot(opt, npy_item['gt_local_pose'])

            values_dict = {}
            for key in ['gt_local_pose', 'gt_root_translation', 'gt_root_translation_delta', # 'gt_camera_frame_pose',
                        'gt_root_rotation',]:
                if npy_item[key] is not None:
                    values_dict[key] = torch.tensor(npy_item[key])

            stat_dict.update(values_dict)

            npy_new_path = os.path.join(npy_new_dir, tail)
            npy_new_data = np.array(npy_item, dtype=object)
            if not opt.experiment:
                np.save(npy_new_path, npy_new_data, allow_pickle=True)

    # Write std and mean for each data in a pickle
    meanstd = stat_dict.get_value()
    meanstd = {k: (v[0].numpy(), v[1].numpy()) for k, v in meanstd.items()}
    for k, v in meanstd.items():
        std = v[1]
        std[std < 0.001] = 1.0

    meanstd_pkl_path = os.path.join(opt.data_dir, 'meanstd')
    if not opt.experiment:
        np.save(meanstd_pkl_path, meanstd, allow_pickle=True)

    print("Failed to modify {} files".format(fail_cnt))
    fail_file.close()
    
    
def is_rotation(mat):
    return np.isclose(np.linalg.det(mat), 1.0)


if __name__ == "__main__":
    opt = DatasetOptions().parse()
    modify_dataset(opt)
