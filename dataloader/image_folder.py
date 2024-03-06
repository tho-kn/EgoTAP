import os, glob
import os.path
from tqdm import tqdm
from natsort import natsorted


def make_dataset(opt, data_list_path, data_sub_path, id=None, check_integrity=False, as_sequence=False, use_metadata=False):
    data = []
    sequences = []

    with open(data_list_path) as f:
        paths = [s.strip() for s in f.readlines()]

    missing_sequence = []

    desc = "Making dataset from {}".format(data_list_path)
    if id is not None:
        desc += " for id {}".format(id)
    
    if check_integrity:
        dataset_dir = os.path.dirname(data_list_path)
        dataset_filename = os.path.basename(data_list_path)
        dataset_file = os.path.join(dataset_dir, "inuse_" + dataset_filename)
        dataset_file = open(dataset_file, "w")

    for path in tqdm(paths, total=len(paths), desc=desc):
        orig_path = path
        if not use_metadata:
            path = path.replace(opt.default_data_path, opt.data_dir, 1)
        else:
            for metadir in opt.metadata_dir:
                new_path = path.replace(opt.default_data_path, metadir, 1)
                if os.path.isdir(new_path):
                    path = new_path
                    break

        full_path = os.path.join(path, data_sub_path, "*")

        if id is not None:
            motion_category_id = full_path.split("/")[-4]
            if id != motion_category_id:
                continue

        list_imgs_per_sequence = natsorted(glob.glob(full_path))

        if check_integrity:
            if len(list_imgs_per_sequence) == 0:
                missing_sequence.append(os.path.join(path, data_sub_path))
            elif not use_metadata:
                dataset_file.write(orig_path + "\n")
            for i in range(len(list_imgs_per_sequence)):
                if not os.path.exists(os.path.join(path, data_sub_path, "frame_{}.npy".format(i))):
                    missing_sequence.append(os.path.join(path, data_sub_path))
                    break

        data += list_imgs_per_sequence
        if len(list_imgs_per_sequence) != 0:
            sequences.append(list_imgs_per_sequence)

        if opt.experiment:
            if len(data) >= 100:
                data = data[:100]
                break
    
    if check_integrity:
        dataset_file.close()

    if as_sequence:
        data = sequences
    ret_val = [data, len(data)]

    if check_integrity:
        ret_val.append(missing_sequence)

    return tuple(ret_val)
