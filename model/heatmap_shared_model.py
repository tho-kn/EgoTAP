import torch
from torch.cuda.amp import autocast, GradScaler
from torch.nn import MSELoss

from .base_model import BaseModel
from . import network


class HeatmapSharedModel(BaseModel):
    def name(self):
        return 'Heatmap Shared model'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        self.opt = opt
        self.scaler = GradScaler(enabled=opt.use_amp)

        self.loss_names = []

        self.visual_names = [
            'input_rgb_left', 'input_rgb_right',
        ]
        
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

        self.visual_pose_names = [
        ]
       
        if self.isTrain:
            self.model_names = ['HeatMap']
        else:
            self.model_names = ['HeatMap']

        self.eval_key = "mse_heatmap"
        self.cm2mm = 10


        # define the transform network
        print(opt.model)
        self.net_HeatMap = network.define_HeatMap(opt, model=opt.model)

        if self.isTrain:
            if opt.path_to_trained_heatmap is not None:
                self.load_networks(
                    net=self.net_HeatMap, 
                    path_to_trained_weights=opt.path_to_trained_heatmap
                )
            
            # define loss functions
            self.lossfunc_MSE = MSELoss()

            # initialize optimizers
            self.optimizer_HeatMap = torch.optim.Adam(
                params=self.net_HeatMap.parameters(), 
                lr=opt.lr,
                weight_decay=opt.weight_decay
            )

            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer_HeatMap)
            for optimizer in self.optimizers:
                self.schedulers.append(network.get_scheduler(optimizer, opt))


    def set_input(self, data):
        self.data = data
        self.input_rgb_left = data['input_rgb_left'].cuda(self.device)
        self.input_rgb_right = data['input_rgb_right'].cuda(self.device)
        self.gt_heatmap_left = data['gt_heatmap_left'].cuda(self.device)
        self.gt_heatmap_right = data['gt_heatmap_right'].cuda(self.device)
        self.gt_pose = data['gt_local_pose'].cuda(self.device)
        self.gt_limb_theta = data['gt_limb_theta'].cuda(self.device)
        
        if self.opt.num_rot_heatmap > 0:
            self.gt_limb_heatmap_left = data['gt_limb_heatmap_left'].cuda(self.device)
            self.gt_limb_heatmap_right = data['gt_limb_heatmap_right'].cuda(self.device)
            self.gt_plength_left = data['gt_plength_left'].cuda(self.device)
            self.gt_plength_right = data['gt_plength_right'].cuda(self.device)
        
    def forward(self):
        with autocast(enabled=self.opt.use_amp):
            # estimate stereo heatmaps
            if self.opt.stereo:
                self.pred_heatmap_cat = self.net_HeatMap(self.input_rgb_left, self.input_rgb_right)
                self.pred_heatmap_left, self.pred_heatmap_right = torch.chunk(self.pred_heatmap_cat[:, :self.opt.num_heatmap*2], 2, dim=1)
                self.pred_limb_heatmap_left, self.pred_limb_heatmap_right = torch.chunk(self.pred_heatmap_cat[:, self.opt.num_heatmap*2:], 2, dim=1)
            else:
                self.pred_heatmap_cat = self.net_HeatMap(self.input_rgb_left)
                self.pred_heatmap_left = self.pred_heatmap_cat[:, :self.opt.num_heatmap]
                self.pred_limb_heatmap_left = self.pred_heatmap_cat[:, self.opt.num_heatmap:]

    def backward_HeatMap(self):
        with autocast(enabled=self.opt.use_amp):
            loss_total = 0
            
            if self.opt.num_heatmap > 0:
                loss_heatmap_left = self.lossfunc_MSE(
                    self.pred_heatmap_left, self.gt_heatmap_left
                )
                self.loss_heatmap_left = loss_heatmap_left * self.opt.lambda_heatmap
                
                loss_total += self.loss_heatmap_left
                
                if self.opt.stereo:
                    loss_heatmap_right = self.lossfunc_MSE(
                        self.pred_heatmap_right, self.gt_heatmap_right
                    )
                    self.loss_heatmap_right = loss_heatmap_right * self.opt.lambda_heatmap

                    loss_total += self.loss_heatmap_right
            
            if self.opt.num_rot_heatmap > 0:
                gt_sqrt_limb_length_left = torch.sqrt(self.gt_plength_left[..., None, None])
                
                norm_pred_limb_heatmap_left = self.pred_limb_heatmap_left / gt_sqrt_limb_length_left
                norm_gt_limb_heatmap_left = self.gt_limb_heatmap_left / gt_sqrt_limb_length_left
                loss_limb_heatmap_left = self.lossfunc_MSE(
                    norm_pred_limb_heatmap_left, norm_gt_limb_heatmap_left
                )
                self.loss_limb_heatmap_left = loss_limb_heatmap_left * self.opt.lambda_rot_heatmap 
                
                loss_total += self.loss_limb_heatmap_left
                
                if self.opt.stereo:
                    gt_sqrt_limb_length_right = torch.sqrt(self.gt_plength_right[..., None, None])
                    norm_pred_limb_heatmap_right = self.pred_limb_heatmap_right / gt_sqrt_limb_length_right
                    norm_gt_limb_heatmap_right = self.gt_limb_heatmap_right / gt_sqrt_limb_length_right
                    loss_limb_heatmap_right = self.lossfunc_MSE(
                        norm_pred_limb_heatmap_right, norm_gt_limb_heatmap_right
                    )
                    self.loss_limb_heatmap_right = loss_limb_heatmap_right * self.opt.lambda_rot_heatmap

                    loss_total += self.loss_limb_heatmap_right
                    
        self.scaler.scale(loss_total).backward()

    def optimize_parameters(self):

        # set model trainable
        self.net_HeatMap.train()
        
        # set optimizer.zero_grad()
        self.optimizer_HeatMap.zero_grad()

        # forward
        self.forward()

        # backward 
        self.backward_HeatMap()

        # optimizer step
        self.scaler.step(self.optimizer_HeatMap)

        self.scaler.update()

    def evaluate(self, runnning_average_dict):
        # set evaluation mode
        self.net_HeatMap.eval()

        # forward pass
        self.forward()

        # compute metrics
        for id in range(self.pred_heatmap_left.size()[0]):  # batch size
            mse_heatmap = 0
            if self.opt.num_heatmap > 0:
                # calculate mse loss for heatmap
                loss_heatmap_left_id = self.lossfunc_MSE(
                    self.pred_heatmap_left[id], self.gt_heatmap_left[id]
                )
                mse_heatmap += loss_heatmap_left_id
                
                if self.opt.stereo:
                    loss_heatmap_right_id = self.lossfunc_MSE(
                        self.pred_heatmap_right[id], self.gt_heatmap_right[id]
                    )
                
                    mse_heatmap += loss_heatmap_right_id

            if self.opt.num_rot_heatmap > 0:
                gt_sqrt_limb_length_left = torch.sqrt(self.gt_plength_left[..., None, None])
                loss_limb_heatmap_left_id = self.lossfunc_MSE(
                    self.pred_limb_heatmap_left[id] / gt_sqrt_limb_length_left[id],
                    self.gt_limb_heatmap_left[id] / gt_sqrt_limb_length_left[id],
                )
                mse_heatmap += loss_limb_heatmap_left_id
                
                if self.opt.stereo:
                    gt_sqrt_limb_length_right = torch.sqrt(self.gt_plength_right[..., None, None])
                    loss_limb_heatmap_right_id = self.lossfunc_MSE(
                        self.pred_limb_heatmap_right[id] / gt_sqrt_limb_length_right[id],
                        self.gt_limb_heatmap_right[id] / gt_sqrt_limb_length_right[id],
                    )
                    mse_heatmap += loss_limb_heatmap_right_id
                
            # update metrics dict
            runnning_average_dict.update(dict(
                mse_heatmap=mse_heatmap
                )
            )

        return None, self.pred_heatmap_cat, runnning_average_dict