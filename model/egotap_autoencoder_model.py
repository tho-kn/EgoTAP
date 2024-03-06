import torch
from torch.cuda.amp import autocast, GradScaler
from torch.nn import MSELoss

from .base_pose_pred_model import BasePosePredModel as BaseModel
from . import network
from utils.loss import LossFuncLimb, LossFuncCosSim, LossFuncMPJPE
from utils.util import batch_compute_similarity_transform_torch
import os
import copy


class EgoTAPAutoEncoderModel(BaseModel):
    def name(self):
        return 'EgoTAP AutoEncoder model'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        self.opt = opt
        self.scaler = GradScaler(enabled=opt.use_amp)

        self.loss_names = []

        if self.isTrain:
            self.visual_names = [
                'input_rgb_left', 'input_rgb_right',
            ]
        else:
            self.visual_names = []
        
        self.train_heatmap = self.isTrain and opt.path_to_trained_heatmap is None
            
        if self.isTrain:
            self.loss_names += ['pose', 'cos_sim',]
            
            if self.isTrain:
                self.visual_names.extend([
                    'gt_heatmap_left', 'gt_heatmap_right',
                ])
            self.visual_names.extend([
                'pred_heatmap_left', 'pred_heatmap_right',
            ])
            
            if self.isTrain:
                self.visual_names.extend([
                    'gt_limb_heatmap_left', 'gt_limb_heatmap_right',
                ])
            self.visual_names.extend([
                'pred_limb_heatmap_left', 'pred_limb_heatmap_right',
            ])
                
            if self.train_heatmap:
                if opt.num_heatmap > 0:
                    self.loss_names.extend([
                        'heatmap_left', 'heatmap_right',
                    ])
                    self.visual_names.extend([
                        'pred_heatmap_left', 'pred_heatmap_right',
                        'gt_heatmap_left', 'gt_heatmap_right',
                    ])
                    
                if opt.num_rot_heatmap > 0:
                    self.loss_names.extend([
                        'limb_heatmap_left', 'limb_heatmap_right',
                    ])
                    self.visual_names.extend([
                        'pred_limb_heatmap_left', 'pred_limb_heatmap_right',
                        'gt_limb_heatmap_left', 'gt_limb_heatmap_right',
                    ])

        if self.train_heatmap:
            if opt.num_heatmap > 0:
                self.loss_names.extend([
                    'heatmap_left', 'heatmap_right',
                ])
                
            if opt.num_rot_heatmap > 0:
                self.loss_names.extend([
                    'limb_heatmap_left', 'limb_heatmap_right',
                ])
                
        if not self.opt.stereo:
            new_loss_name = []
            for name in self.loss_names:
                if not name.endswith("_right"):
                    new_loss_name.append(name)
            self.loss_names = new_loss_name

        self.visual_pose_names = [
            "pred_pose", "gt_pose"
        ]
       
        if self.isTrain:
            self.model_names = ['HeatMap', 'RotHeatMap', 'AutoEncoder']
        else:
            self.model_names = ['HeatMap', 'RotHeatMap', 'AutoEncoder']

        self.eval_key = "mpjpe"
        self.cm2mm = 10
        self.stereo = opt.stereo

        # define the transform network
        pos_opt = copy.deepcopy(opt)
        pos_opt.num_rot_heatmap = 0
        rot_opt = copy.deepcopy(opt)
        rot_opt.num_heatmap = 0
        
        self.net_HeatMap = network.define_HeatMap(pos_opt, model=opt.model)
        self.net_RotHeatMap = network.define_HeatMap(rot_opt, model=opt.model)
        self.net_AutoEncoder = network.define_AutoEncoder(opt, model=opt.model)

        if self.isTrain and (not self.train_heatmap or opt.path_to_trained_heatmap is not None):
            pretrained_path = opt.path_to_trained_heatmap
            pretrained_dir = os.path.dirname(pretrained_path)
            pretrained_file = os.path.basename(pretrained_path)
            self.load_networks(
                net=self.net_HeatMap, 
                path_to_trained_weights=os.path.join(pretrained_dir + "_pos", pretrained_file)
            )
            net_type = self.opt.heatmap_type
                
            self.load_networks(
                net=self.net_RotHeatMap, 
                path_to_trained_weights=os.path.join(pretrained_dir + "_" + net_type, pretrained_file)
            )
            if not self.train_heatmap:
                network._freeze(self.net_HeatMap)
                network._freeze(self.net_RotHeatMap)
        
        self.input_channel_scale = 2 if opt.stereo else 1
    
        # define loss functions
        self.lossfunc_MSE = MSELoss()
        self.lossfunc_limb = LossFuncLimb(opt.joint_preset)
        self.lossfunc_cos_sim = LossFuncCosSim(joint_preset=opt.joint_preset, estimate_head=opt.estimate_head)
        self.lossfunc_MPJPE = LossFuncMPJPE()

        if self.isTrain:
            # initialize optimizers
            self.optimizers = []
            self.schedulers = []
            
            self.optimizer_AutoEncoder = network.get_optimizer(
                params=self.net_AutoEncoder.parameters(),
                opt=self.opt,
            )
            self.optimizers.append(self.optimizer_AutoEncoder)
            

            for optimizer in self.optimizers:
                self.schedulers.append(network.get_scheduler(optimizer, opt))


    def set_input(self, data):
        self.data = data
        self.input_rgb_left = data['input_rgb_left'].cuda(self.device)
        self.input_rgb_right = data['input_rgb_right'].cuda(self.device)
        self.gt_heatmap_left = data['gt_heatmap_left'].cuda(self.device)
        self.gt_heatmap_right = data['gt_heatmap_right'].cuda(self.device)
        self.gt_pose = data['gt_local_pose'].cuda(self.device)
        self.gt_rot = data['gt_local_rot'][..., -self.opt.num_rot_heatmap:, :].cuda(self.device).view(-1, self.opt.num_rot_heatmap * 3)
        self.gt_limb_theta = data['gt_limb_theta'].cuda(self.device)
        
        self.gt_pelvis_left = data['gt_pelvis_left'].cuda(self.device)
        self.gt_pelvis_right =  data['gt_pelvis_right'].cuda(self.device)
        
        batch_dim = len(self.input_rgb_left.shape) - 3
        self.gt_pelvis = torch.stack((self.gt_pelvis_left, self.gt_pelvis_right), dim=batch_dim)

        self.gt_limb_heatmap_left = data['gt_limb_heatmap_left'].cuda(self.device)
        self.gt_limb_heatmap_right = data['gt_limb_heatmap_right'].cuda(self.device)
        self.gt_plength_left = data['gt_plength_left'].cuda(self.device)
        self.gt_plength_right = data['gt_plength_right'].cuda(self.device)
        
        
    def forward_heatmap(self):
        # estimate stereo heatmaps
        with torch.set_grad_enabled(self.train_heatmap):
            if self.stereo:
                if self.opt.use_gt_heatmap:
                    pred_heatmap_cat = torch.cat((self.gt_heatmap_left, self.gt_heatmap_right), dim=1)
                    pred_limb_heatmap_cat = torch.cat((self.gt_limb_heatmap_left, self.gt_limb_heatmap_right), dim=1)
                else:
                    pred_heatmap_cat = self.net_HeatMap(self.input_rgb_left, self.input_rgb_right)
                    pred_limb_heatmap_cat = self.net_RotHeatMap(self.input_rgb_left, self.input_rgb_right)
            else:
                if self.opt.use_gt_heatmap:
                    pred_heatmap_cat = self.gt_heatmap_left
                    pred_limb_heatmap_cat = self.gt_limb_heatmap_left
                else:
                    pred_heatmap_cat = self.net_HeatMap(self.input_rgb_left)
                    pred_limb_heatmap_cat = self.net_RotHeatMap(self.input_rgb_left)
            
            if self.stereo:
                self.pred_heatmap_left, self.pred_heatmap_right = torch.chunk(pred_heatmap_cat, 2, dim=1)
            else:
                self.pred_heatmap_left = pred_heatmap_cat
                self.pred_heatmap_right = pred_heatmap_cat
                
            if self.input_channel_scale == 1:
                pred_heatmap_cat = self.pred_heatmap_left
                self.pred_heatmap_right = self.pred_heatmap_left
            
            if self.stereo:
                self.pred_limb_heatmap_left, self.pred_limb_heatmap_right = torch.chunk(pred_limb_heatmap_cat, 2, dim=1)
            else:
                self.pred_limb_heatmap_left = pred_limb_heatmap_cat
                self.pred_limb_heatmap_right = pred_limb_heatmap_cat
                
            if self.input_channel_scale == 1:
                pred_limb_heatmap_cat = self.pred_limb_heatmap_left
                self.pred_limb_heatmap_right = self.pred_limb_heatmap_left
            
            pred_heatmap_cat = torch.cat((pred_heatmap_cat, pred_limb_heatmap_cat), dim=1)
        self.pred_heatmap_cat = pred_heatmap_cat
            
    def forward(self, evaluate=False):
        with autocast(enabled=self.opt.use_amp and not evaluate):
            # estimate pose and reconstruct stereo heatmaps
            self.forward_heatmap()
            
            self.pred_pose, self.pred_rot, self.pred_indep_pos, pred_heatmap_rec_cat = self.net_AutoEncoder(self.pred_heatmap_cat, self.input_rgb_left, self.input_rgb_right)
            pred_heatmap_recs  = torch.chunk(pred_heatmap_rec_cat[:, :self.opt.num_heatmap*self.input_channel_scale], self.input_channel_scale, dim=1)
            self.pred_heatmap_rec_cat = pred_heatmap_rec_cat
            if self.input_channel_scale == 1:
                self.pred_heatmap_left_rec = pred_heatmap_recs[0]
                self.pred_heatmap_right_rec = pred_heatmap_recs[0]
            else:
                self.pred_heatmap_left_rec, self.pred_heatmap_right_rec = pred_heatmap_recs
            
            pred_limb_heatmap_recs = torch.chunk(pred_heatmap_rec_cat[:, self.opt.num_heatmap*self.input_channel_scale:], self.input_channel_scale, dim=1)
            if self.input_channel_scale == 1:
                self.pred_limb_heatmap_left_rec = pred_limb_heatmap_recs[0]
                self.pred_limb_heatmap_right_rec = pred_limb_heatmap_recs[0]
            else:
                self.pred_limb_heatmap_left_rec, self.pred_limb_heatmap_right_rec = pred_limb_heatmap_recs
    
    def backward_HeatMap(self):
        with autocast(enabled=self.opt.use_amp):
            heatmap_losses = []
            if self.opt.num_heatmap > 0:
                loss_heatmap_left = self.lossfunc_MSE(
                    self.pred_heatmap_left, self.gt_heatmap_left
                )
                self.loss_heatmap_left = loss_heatmap_left * self.opt.lambda_heatmap
                
                heatmap_losses.append(self.loss_heatmap_left)
                
                if self.stereo:
                    loss_heatmap_right = self.lossfunc_MSE(
                        self.pred_heatmap_right, self.gt_heatmap_right
                    )
                    self.loss_heatmap_right = loss_heatmap_right * self.opt.lambda_heatmap

                    heatmap_losses.append(self.loss_heatmap_right)
            
            if self.opt.num_rot_heatmap > 0:
                gt_sqrt_limb_length_left = torch.sqrt(self.gt_plength_left[..., None, None])
                
                norm_pred_limb_heatmap_left = self.pred_limb_heatmap_left / gt_sqrt_limb_length_left
                norm_gt_limb_heatmap_left = self.gt_limb_heatmap_left / gt_sqrt_limb_length_left
                loss_limb_heatmap_left = self.lossfunc_MSE(
                    norm_pred_limb_heatmap_left, norm_gt_limb_heatmap_left
                )
                self.loss_limb_heatmap_left = loss_limb_heatmap_left * self.opt.lambda_rot_heatmap 
                
                heatmap_losses.append(self.loss_limb_heatmap_left)
                
                if self.stereo:
                    gt_sqrt_limb_length_right = torch.sqrt(self.gt_plength_right[..., None, None])
                    norm_pred_limb_heatmap_right = self.pred_limb_heatmap_right / gt_sqrt_limb_length_right
                    norm_gt_limb_heatmap_right = self.gt_limb_heatmap_right / gt_sqrt_limb_length_right
                    loss_limb_heatmap_right = self.lossfunc_MSE(
                        norm_pred_limb_heatmap_right, norm_gt_limb_heatmap_right
                    )
                    self.loss_limb_heatmap_right = loss_limb_heatmap_right * self.opt.lambda_rot_heatmap

                    heatmap_losses.append(self.loss_limb_heatmap_right)
            
            for loss in heatmap_losses:
                self.loss_total += loss

    def backward_AutoEncoder(self):
        with autocast(enabled=self.opt.use_amp):
            losses = []
            loss_pose = self.lossfunc_MPJPE(self.pred_pose, self.gt_pose)
            loss_cos_sim = self.lossfunc_cos_sim(self.pred_pose, self.gt_pose)

            self.loss_pose = loss_pose * self.opt.lambda_mpjpe
            self.loss_cos_sim = loss_cos_sim * self.opt.lambda_cos_sim * self.opt.lambda_mpjpe
            
            losses.extend([self.loss_pose, self.loss_cos_sim,])
            
            for loss in losses:
                    self.loss_total += loss


    def optimize_parameters(self):
        # set model trainable
        self.net_AutoEncoder.train()
        
        # set optimizer.zero_grad()
        for optimizer in self.optimizers:
            optimizer.zero_grad()

        # forward
        self.forward()

        self.loss_total = 0.0
        
        # backward 
        if self.train_heatmap:
            self.backward_HeatMap()
        self.backward_AutoEncoder()
        
        self.scaler.scale(self.loss_total).backward()
        
        # optimizer step
        for optimizer in self.optimizers:
            self.scaler.step(optimizer)

        self.scaler.update()
        
    def set_eval_mode(self):
        self.net_AutoEncoder.eval()
        self.net_HeatMap.eval()

    def evaluate(self, runnning_average_dict):
        # set evaluation mode
        self.set_eval_mode()

        with torch.no_grad():
            self.forward(evaluate=True)

        S1_hat = batch_compute_similarity_transform_torch(self.pred_pose, self.gt_pose)

        # compute metrics
        for id in range(self.pred_pose.size()[0]):  # batch size
            # calculate mpjpe and p_mpjpe   # cm to mm
            mpjpe = self.lossfunc_MPJPE(self.pred_pose[id], self.gt_pose[id]) * self.cm2mm
            pa_mpjpe = self.lossfunc_MPJPE(S1_hat[id], self.gt_pose[id]) * self.cm2mm
            
            metrics = dict(
                mpjpe=mpjpe, 
                pa_mpjpe=pa_mpjpe)
            # update metrics dict
            runnning_average_dict.update(metrics)

        return self.pred_pose, self.pred_heatmap_cat, runnning_average_dict


