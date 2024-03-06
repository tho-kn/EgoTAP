import numpy as np
import torch
import os
from PIL import Image
import json
from copy import deepcopy
from matplotlib import pyplot as plt

# Joint hierarchy for UnrealEgo
index_to_ue_joint_name = {
    0: "head",
    1: "neck_01",
    2: "upperarm_l",
    3: "upperarm_r",
    4: "lowerarm_l",
    5: "lowerarm_r",
    6: "hand_l",
    7: "hand_r",
    8: "thigh_l",
    9: "thigh_r",
    10: "calf_l",
    11: "calf_r",
    12: "foot_l",
    13: "foot_r",
    14: "ball_l",
    15: "ball_r"
}

# Head is not used for evaluation
index_to_egocap_joint_name = {
    0: "cam",
    1: "neck",
    2: "left_shoulder",
    3: "left_elbow",
    4: "left_wrist",
    5: "left_finger",
    6: "right_shoulder",
    7: "right_elbow",
    8: "right_wrist",
    9: "right_finger",
    10: "left_hip",
    11: "left_knee",
    12: "left_ankle",
    13: "left_toe",
    14: "right_hip",
    15: "right_knee",
    16: "right_ankle",
    17: "right_toe"
}

ue_kinematic_parents = [0, 0, 1, 1, 2, 3, 4, 5, 2, 3, 8, 9, 10, 11, 12, 13]
egocap_kinematic_parents = [0, 0, 1, 2, 3, 4, 1, 6, 7, 8, 2, 10, 11, 12, 6, 14, 15, 16]

def get_index_to_joint_name(joint_preset):
    if joint_preset == 'UnrealEgo':
        return index_to_ue_joint_name
    if joint_preset == 'EgoCap':
        return index_to_egocap_joint_name
    raise ValueError("joint_preset is {} which is undefined".format(joint_preset))
    
def get_kinematic_parents(joint_preset):
    if joint_preset == 'UnrealEgo':
        return ue_kinematic_parents
    if joint_preset == 'EgoCap':
        return egocap_kinematic_parents
    raise ValueError("joint_preset is {} which is undefined".format(joint_preset))


def print_current_errors(epoch, i, errors, t):
    message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)
    for k, v in errors.items():
        if abs(v) > 1e-1:
            message += '{}: {:.3f} '.format(k, v.item())
        else:
            message += '{}: {:.3e} '.format(k, v.item())

    print(message)

class RunningAverage:
    def __init__(self):
        self.initialized = False

    def append(self, value, avg_dim=0):
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value)
        value = torch.flatten(value, start_dim=0, end_dim=avg_dim)
        if not self.initialized:
            if avg_dim == 0:
                self.avg = value
                self.count = torch.tensor(1)
            else:
                self.avg = torch.mean(value, dim=0)
                self.count = torch.tensor(value.shape[0])
            self.initialized = True
        else:
            self.avg = (value + self.count * self.avg) / (self.count + 1)
            self.count += 1

    def get_value(self):
        return self.avg

class RunningAverageStd:
    def __init__(self):
        self.initialized = False

    def append(self, value, avg_dim=0):
        value = torch.flatten(value, start_dim=0, end_dim=avg_dim)
        if not self.initialized:
            if avg_dim == 0:
                self.avg = value
                self.M2 = torch.zeros_like(value)
                self.count = torch.tensor(1)
            else:
                self.avg = torch.mean(value, dim=0)
                self.M2 = torch.var(value, dim=0) * value.shape[0]
                self.count = torch.tensor(value.shape[0])
            self.initialized = True
        else:
            self.count += 1
            delta = value - self.avg
            self.avg += delta / self.count
            delta2 = value - self.avg
            self.M2 += delta * delta2

    def get_value(self):
        if self.count < 2:
            raise ValueError("Variance is undefined for less than 2 values")
        else:
            std = torch.sqrt(self.M2 / (self.count - 1))
            return (self.avg, std)

class RunningDict:
    def __init__(self):
        self._dict = None
    
    def new_dict(self):
        return []

    def update(self, new_dict):
        if self._dict is None:
            self._dict = dict()
            for key, value in new_dict.items():
                self._dict[key] = self.new_dict()

        for key, value in new_dict.items():
            self._dict[key].append(value)

    def get_value(self):
        return {key: value.get_value() for key, value in self._dict.items()}

class RunningAverageDict(RunningDict):
    def new_dict(self):
        return RunningAverage()

class RunningAverageStdDict(RunningDict):
    def new_dict(self):
        return RunningAverageStd()

# convert a tensor into a numpy array
def tensor2im(image_tensor, bytes=255.0, imtype=np.uint8, is_depth=False, is_heatmap=False):
    if image_tensor.dim() == 3:  # (C, H, W)
        image_tensor = image_tensor.cpu().float()
    else:
        image_tensor = image_tensor[0].cpu().float()
    
    if is_depth:
        image_tensor = image_tensor * bytes
    elif is_heatmap:
        image_tensor = torch.clamp(torch.sum(image_tensor, dim=0, keepdim=True), min=0.0, max=1.0) * bytes
    else:
        # image_tensor = (image_tensor + 1.0) / 2.0 * bytes
        image_tensor = denormalize_ImageNet(image_tensor) * bytes

    image_numpy = (image_tensor.permute(1, 2, 0)).numpy().astype(imtype)
    return image_numpy

def denormalize_ImageNet(x):
    if isinstance(x, torch.Tensor):
        mean = torch.Tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.Tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    if isinstance(x, np.ndarray):
        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    x = x * std
    x = x + mean
    return x

def normalize_ImageNet(x):
    if isinstance(x, torch.Tensor):
        mean = torch.Tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.Tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    if isinstance(x, np.ndarray):
        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    x = x - mean
    x = x / std
    return x

def save_image(image_numpy, image_path):
    if image_numpy.shape[2] == 1:
        image_numpy = image_numpy.reshape(image_numpy.shape[0], image_numpy.shape[1])
    image_numpy = Image.fromarray(image_numpy)
    image_numpy.save(image_path)
    # imageio.imwrite(image_path, image_numpy)

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Some functions are borrowed from https://github.com/akanazawa/human_dynamics/blob/master/src/evaluation/eval_util.py
# Adhere to their licence to use these functions
def compute_similarity_transform(S1, S2):
    '''
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    '''
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        transposed = True
    assert(S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = np.sum(X1**2)

    # 3. The outer product of X1 and X2.
    K = X1.dot(X2.T)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, Vh = np.linalg.svd(K)
    V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    # Construct R.
    R = V.dot(Z.dot(U.T))

    # 5. Recover scale.
    scale = np.trace(R.dot(K)) / var1

    # 6. Recover translation.
    t = mu2 - scale*(R.dot(mu1))

    # 7. Error:
    S1_hat = scale*R.dot(S1) + t

    if transposed:
        S1_hat = S1_hat.T

    return S1_hat


def compute_similarity_transform_torch(S1, S2):
    '''
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    '''
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        transposed = True
    assert (S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # print('X1', X1.shape)

    # 2. Compute variance of X1 used for scale.
    var1 = torch.sum(X1 ** 2)

    # print('var', var1.shape)

    # 3. The outer product of X1 and X2.
    K = X1.mm(X2.T)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, V = torch.svd(K)
    # V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = torch.eye(U.shape[0], device=S1.device)
    Z[-1, -1] *= torch.sign(torch.det(U @ V.T))
    # Construct R.
    R = V.mm(Z.mm(U.T))

    # print('R', X1.shape)

    # 5. Recover scale.
    scale = torch.trace(R.mm(K)) / var1
    # print(R.shape, mu1.shape)
    # 6. Recover translation.
    t = mu2 - scale * (R.mm(mu1))
    # print(t.shape)

    # 7. Error:
    S1_hat = scale * R.mm(S1) + t

    if transposed:
        S1_hat = S1_hat.T

    return S1_hat


def batch_compute_similarity_transform_torch(S1, S2):
    '''
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    '''
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.permute(0,2,1)
        S2 = S2.permute(0,2,1)
        transposed = True
    assert(S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = S1.mean(axis=-1, keepdims=True)
    mu2 = S2.mean(axis=-1, keepdims=True)

    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = torch.sum(X1**2, dim=1).sum(dim=1)

    # 3. The outer product of X1 and X2.
    K = X1.bmm(X2.permute(0,2,1))

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, V = torch.svd(K)

    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = torch.eye(U.shape[1], device=S1.device).unsqueeze(0)
    Z = Z.repeat(U.shape[0],1,1)
    Z[:,-1, -1] *= torch.sign(torch.det(U.bmm(V.permute(0,2,1))))

    # Construct R.
    R = V.bmm(Z.bmm(U.permute(0,2,1)))

    # 5. Recover scale.
    scale = torch.cat([torch.trace(x).unsqueeze(0) for x in R.bmm(K)]) / var1

    # 6. Recover translation.
    t = mu2 - (scale.unsqueeze(-1).unsqueeze(-1) * (R.bmm(mu1)))

    # 7. Error:
    S1_hat = scale.unsqueeze(-1).unsqueeze(-1) * R.bmm(S1) + t

    if transposed:
        S1_hat = S1_hat.permute(0,2,1)

    return S1_hat


def align_by_pelvis(joints):
    """
    Assumes joints is 14 x 3 in LSP order.
    Then hips are: [3, 2]
    Takes mid point of these points, then subtracts it.
    """

    left_id = 2
    right_id = 3

    pelvis = (joints[left_id, :] + joints[right_id, :]) / 2.0
    return joints - np.expand_dims(pelvis, axis=0)


def compute_errors(gt3ds, preds):
    """
    Gets MPJPE after pelvis alignment + MPJPE after Procrustes.
    Evaluates on the 14 common joints.
    Inputs:
      - gt3ds: N x 14 x 3
      - preds: N x 14 x 3
    """
    errors, errors_pa = [], []
    for i, (gt3d, pred) in enumerate(zip(gt3ds, preds)):
        gt3d = gt3d.reshape(-1, 3)
        # Root align.
        gt3d = align_by_pelvis(gt3d)
        pred3d = align_by_pelvis(pred)

        joint_error = np.sqrt(np.sum((gt3d - pred3d)**2, axis=1))
        errors.append(np.mean(joint_error))

        # Get PA error.
        pred3d_sym = compute_similarity_transform(pred3d, gt3d)
        pa_error = np.sqrt(np.sum((gt3d - pred3d_sym)**2, axis=1))
        errors_pa.append(np.mean(pa_error))

    return errors, errors_pa
    
def try_json(json_path):
    try:
        with open(json_path, "r") as f:
            json_data = json.load(f)
        return json_data
    except:
        return None
    

def denormalize_input_img(inputs):
    if isinstance(inputs, torch.Tensor):
        inputs = inputs.numpy()
    inputs = denormalize_ImageNet(inputs)
    inputs = np.around(inputs*255).astype(np.uint8)
    return inputs

def normalize_input_img(inputs):
    inputs = inputs.astype(np.float32) / 255.0
    inputs = normalize_ImageNet(input)
    return inputs

def get_normalized_image(heatmap):
    heatmap = heatmap / np.max(heatmap) * 255
    if len(heatmap.shape) == 2:
        im = Image.fromarray(heatmap.astype(np.uint8), 'L')
    elif len(heatmap.shape) == 3 and heatmap.shape[2] == 3:
        im = Image.fromarray(heatmap.astype(np.uint8))
    return im


def save_input_img(inputs, path='./results/rgb.png'):
    if isinstance(inputs, np.ndarray):
        inputs = torch.tensor(inputs)
    im = denormalize_ImageNet(inputs)
    im = np.around(im.numpy() * 255.0).astype(np.uint8)
    im = np.moveaxis(im, [0], [2])
    im = Image.fromarray(im)
    im.format = "PNG"
    im.save(path)
    