# TH
import time
import numpy as np
import os
from options.file_check_options import FileCheckOptions
from dataloader.image_folder import make_dataset
from tqdm import tqdm
from PIL import Image
import json
from utils.evaluate import get_dict_motion_category
from IPython import embed


# Check pickles if it loads correctly and write down list of corrupted pickles
if __name__ == '__main__':
    opt = FileCheckOptions().parse()
    path = opt.data_dir

    modes = ['train', 'test', 'validation']
    
    all_frame_data_paths, all_num_frame_data = [], 0
    corrupt_sequences = []
    all_num_frame_data_per_category = [0] * len(get_dict_motion_category())
    
    for mode in modes:
        for key, value in get_dict_motion_category().items():
            data_list_path = os.path.join(opt.data_dir, mode + '.txt')  # CT\UnrealEgo\static00\UnrealEgoData\train.txt
            dataset = make_dataset(
                opt=opt,
                data_list_path=data_list_path, 
                data_sub_path=opt.data_sub_path,
                id=key,
                check_integrity=opt.check_integrity,
                use_metadata=True,
            )

            if opt.check_integrity:
                frame_data_paths, num_frame_data, missing_sequences = dataset
                corrupt_sequences.extend(missing_sequences)
            else:
                frame_data_paths, num_frame_data = dataset

            all_frame_data_paths.extend(frame_data_paths)
            all_num_frame_data += num_frame_data
            all_num_frame_data_per_category[int(key)-1] += num_frame_data

    print("Found {} frame npy data".format(all_num_frame_data))
    for key, value in get_dict_motion_category().items():
        print("Found {} frame npy data for {}".format(all_num_frame_data_per_category[int(key)-1], value))

    if opt.check_integrity:
        with open(os.path.join(opt.data_dir, "corrupt_sequence.txt"), "w") as sequence_files:
            for filename in sorted(corrupt_sequences):
                sequence_files.write(filename + "\n")
        print("Found {} missing sequences.".format(len(corrupt_sequences)))

    corrupt_pickles = []
    num_corrupt_pickles = 0

    corrupt_json = []
    corrupt_depth = []
    corrupt_rgb = []

    with open(os.path.join(opt.data_dir, "dataset_check_log.txt"), "w") as log_file:
        for pkl_path in tqdm(all_frame_data_paths, total=len(all_frame_data_paths),
            desc="Checking corrupted or missing files"):
            try:
                frame_data = np.load(pkl_path, allow_pickle=True)
            except Exception as e:
                log_file.write(pkl_path + ": Corrupted Pickle. " + str(e) + "\n")
                corrupt_pickles.append(pkl_path)
                num_corrupt_pickles += 1
                continue

            # Check the rest of the files only if pickle is valid
            [head, tail] = os.path.split(pkl_path)
            take_data_dir = []
            for metadir in opt.metadata_dir:
                take_data_dir = os.path.join(head, os.pardir).replace(opt.data_dir, metadir)
                if os.path.isdir(take_data_dir):
                    break
            if opt.check_json:
                json_path = os.path.join(take_data_dir, 'json', tail[:-4] + ".json")
                try:
                    with open(json_path, "r") as f:
                        json.load(f)
                except Exception as e:
                    corrupt_json.append(json_path)

            if opt.check_depth_image:
                image_dir = os.path.join(take_data_dir, 'fisheye_depth_image')
                left_path = os.path.join(image_dir, 'camera_left', 'depth' + tail[5:-4] + ".png")
                right_path = os.path.join(image_dir, 'camera_right', 'depth' + tail[5:-4] + ".png")
                try:
                    Image.open(left_path).verify()
                except(IOError, SyntaxError) as e:
                    corrupt_depth.append(left_path)
                try:
                    Image.open(right_path).verify()
                except(IOError, SyntaxError) as e:
                    corrupt_depth.append(right_path)

            if opt.check_rgb_image:
                image_dir = os.path.join(take_data_dir, 'fisheye_final_image')
                left_path = os.path.join(image_dir, 'camera_left', 'final' + tail[5:-4] + ".png")
                right_path = os.path.join(image_dir, 'camera_right', 'final' + tail[5:-4] + ".png")
                try:
                    Image.open(left_path).verify()
                except(IOError, SyntaxError) as e:
                    corrupt_rgb.append(left_path)
                try:
                    Image.open(right_path).verify()
                except(IOError, SyntaxError) as e:
                    corrupt_rgb.append(right_path)

    with open(os.path.join(opt.data_dir, "corrupt_npy.txt"), "w") as corrupt_files:
        for filename in sorted(corrupt_pickles):
            corrupt_files.write(filename + "\n")
        print("Found {} corrupted npy files.".format(len(corrupt_pickles)))

    if opt.check_json:
        with open(os.path.join(opt.data_dir, "corrupt_json.txt"), "w") as json_files:
            for filename in sorted(corrupt_json):
                json_files.write(filename + "\n")
        print("Found {} corrupted json files.".format(len(corrupt_json)))

    if opt.check_depth_image:
        with open(os.path.join(opt.data_dir, "corrupt_depth.txt"), "w") as depth_files:
            for filename in sorted(corrupt_depth):
                depth_files.write(filename + "\n")
        print("Found {} corrupted depth images.".format(len(corrupt_depth)))

    if opt.check_rgb_image:
        with open(os.path.join(opt.data_dir, "corrupt_rgb.txt"), "w") as rgb_files:
            for filename in sorted(corrupt_rgb):
                rgb_files.write(filename + "\n")
        print("Found {} corrupted rgb images.".format(len(corrupt_rgb)))
