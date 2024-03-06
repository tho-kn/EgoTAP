import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp
from utils.util import get_kinematic_parents
from utils.projection import world2cam, get_ocam_model

list_left_joints = []
list_center_joints = []
list_right_joints = []
list_upper_joints = []
list_lower_joints = []

def l1_regularizer(model):
    l1_loss = 0
    for param in model.parameters():
        l1_loss += torch.sum(torch.abs(param))
    return l1_loss

class LossFuncLimb(nn.Module):
    def __init__(self, joint_preset=None):
        super(LossFuncLimb, self).__init__()

        self.cos_loss = nn.CosineSimilarity(dim=2)
        self.kinematic_parents = get_kinematic_parents(joint_preset)

    def forward(self, pose_predicted, pose_gt):
        predicted_bone_vector = pose_predicted - pose_predicted[..., self.kinematic_parents, :]
        predicted_bone_vector = predicted_bone_vector[..., 1:, :]
        gt_bone_vector = pose_gt - pose_gt[..., self.kinematic_parents, :]
        gt_bone_vector = gt_bone_vector[..., 1:, :]

        cos_loss = self.cos_loss(predicted_bone_vector, gt_bone_vector)
        cos_loss = torch.mean(torch.sum(cos_loss, dim=1), dim=0)

        predicted_bone_length = torch.norm(predicted_bone_vector, dim=-1)
        gt_bone_length = torch.norm(gt_bone_vector, dim=-1)

        bone_length_loss = torch.mean(torch.sum((predicted_bone_length - gt_bone_length)**2, dim=1), dim=0)

        return cos_loss, bone_length_loss

class LossFuncCosSim(nn.Module):
    def __init__(self, pred_rot=False, joint_preset=None, estimate_head=True):
        super(LossFuncCosSim, self).__init__()

        self.cos_loss = nn.CosineSimilarity(dim=2)
        self.pred_rot = pred_rot
        self.kinematic_parents = get_kinematic_parents(joint_preset)
        self.joint_preset = joint_preset
        self.estimate_head = estimate_head

    def forward(self, pose_predicted, pose_gt):
        if not self.pred_rot:
            if not self.estimate_head:
                pose_predicted = torch.concatenate(
                    (torch.zeros((pose_predicted.shape[0], 1, 3), device=pose_predicted.device), pose_predicted), dim=1
                )
            predicted_bone_vector = pose_predicted - pose_predicted[:, self.kinematic_parents, :]
            predicted_bone_vector = predicted_bone_vector[:, 1:, :]
        else:
            predicted_bone_vector = pose_predicted
        
        if not self.estimate_head:
            pose_gt = torch.concatenate(
                (torch.zeros((pose_gt.shape[0], 1, 3), device=pose_gt.device), pose_gt), dim=1
            )
        gt_bone_vector = pose_gt - pose_gt[:, self.kinematic_parents, :]
        gt_bone_vector = gt_bone_vector[:, 1:, :]

        cos_loss = self.cos_loss(predicted_bone_vector, gt_bone_vector)
        if self.estimate_head == False:
            cos_loss = cos_loss[:, 1:]
        cos_loss = torch.mean(torch.sum(cos_loss, dim=1), dim=0)

        return cos_loss

class LossFuncMPJPE(nn.Module): 
    def __init__(self):
        super(LossFuncMPJPE, self).__init__()

    def forward(self, pred_pose, gt_pose):
        distance = torch.linalg.norm(gt_pose - pred_pose, dim=-1)
        return torch.mean(distance)

if __name__ == "__main__":

    loss = nn.MSELoss(reduction="none")
    input = torch.randn(4, 3, 5, 5, requires_grad=True)
    target = torch.randn(4, 3, 5, 5)
    output = loss(input, target)
    print(output)