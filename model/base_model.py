from operator import contains
import os
import torch
import torch.nn as nn
from collections import OrderedDict
from utils import util
import shutil 


class BaseModel(nn.Module):
    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.save_dir = os.path.join(opt.log_dir, opt.experiment_name)
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.visual_pose_names = []
        self.image_paths = []
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU
    
    def set_input(self, input):
        self.input = input

    # update learning rate
    def update_learning_rate(self):
        old_lr = self.optimizers[0].param_groups[0]['lr']
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        if "anneal" not in self.opt.lr_policy:
            print('learning rate %.7f -> %.7f' % (old_lr, lr))


    # return training loss
    def get_current_errors(self):
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str) and hasattr(self, 'loss_' + name):
                errors_ret[name] = getattr(self, 'loss_' + name).item()
        return errors_ret

    # return visualization images
    def get_current_visuals(self):
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                value = getattr(self, name)

                if "heatmap" in name:
                    is_heatmap = True
                else:
                    is_heatmap = False

                visual_ret[name] = util.tensor2im(value.data, is_heatmap=is_heatmap)

        return visual_ret

    # save models
    def save_networks(self, which_epoch=None, checkpoint_path=None):
        if which_epoch is None and checkpoint_path is None:
            raise ValueError("which_epoch and checkpoint_path cannot be both None")
        
        if which_epoch is None:
            which_epoch = "checkpoint"

        if checkpoint_path is None:
            checkpoint_path = self.save_dir
    
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (which_epoch, name)
                save_path = os.path.join(checkpoint_path, save_filename)
                net = getattr(self, 'net_' + name)
                os.makedirs(checkpoint_path, exist_ok=True)
                torch.save(net.cpu().state_dict(), save_path)
                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    net.cuda()

        for i, optimizer in enumerate(self.optimizers):
            save_optname = '%s_optim_%s.pth' % (which_epoch, i)
            save_path = os.path.join(checkpoint_path, save_optname)
            torch.save(optimizer.state_dict(), save_path)
        
        for i, scheduler in enumerate(self.schedulers):
            save_schname = '%s_scheduler_%s.pth' % (which_epoch, i)
            save_path = os.path.join(checkpoint_path, save_schname)
            torch.save(scheduler.state_dict(), save_path)
                    
        # delete previous files if which epoch is converted to int
        if isinstance(which_epoch, int) and which_epoch > 1:
            prev_epoch = which_epoch - 1
            if which_epoch != self.opt.epoch_count:
                for name in self.model_names:
                    prev_filename = '%s_net_%s.pth' % (prev_epoch, name)
                    prev_path = os.path.join(checkpoint_path, prev_filename)
                    if os.path.exists(prev_path):
                        os.remove(prev_path)
                        
                for i, optimizer in enumerate(self.optimizers):
                    prev_optname = '%s_optim_%s.pth' % (prev_epoch, i)
                    prev_path = os.path.join(checkpoint_path, prev_optname)
                    if os.path.exists(prev_path):
                        os.remove(prev_path)
                        
                for i, scheduler in enumerate(self.schedulers):
                    prev_schname = '%s_scheduler_%s.pth' % (prev_epoch, i)
                    prev_path = os.path.join(checkpoint_path, prev_schname)
                    if os.path.exists(prev_path):
                        os.remove(prev_path)

    # load models
    def load_networks(self, which_epoch=None, net=None, path_to_trained_weights=None, checkpoint_path=None):
        if path_to_trained_weights is None:
            if which_epoch is None and checkpoint_path is None:
                raise ValueError("which_epoch and checkpoint_path cannot be both None")
            if which_epoch is None:
                which_epoch = "checkpoint"
            if checkpoint_path is None:
                checkpoint_path = self.save_dir
            for name in self.model_names:
                print(name)
                if isinstance(name, str):
                    save_filename = '%s_net_%s.pth' % (which_epoch, name)
                    save_path = os.path.join(checkpoint_path, save_filename)
                    net = getattr(self, 'net_'+name)
                    state_dict = torch.load(save_path)
                    net.load_state_dict(state_dict)
                    # net.load_state_dict(self.fix_model_state_dict(state_dict))
                    if not self.isTrain:
                        net.eval()
                    else:
                        net.train()
        else:
            # For compatibility with previous code
            if './log' in path_to_trained_weights:
                path_to_trained_weights = path_to_trained_weights.replace('./log/', '')
            weight_path = os.path.join(self.opt.log_dir, path_to_trained_weights)
            state_dict = torch.load(weight_path)
            if self.opt.distributed:
                net.load_state_dict(self.fix_model_state_dict(state_dict))
            else:
                net.load_state_dict(state_dict)
            print('Loaded pre_trained {}'.format(os.path.basename(weight_path)))
            
    def load_partial_weights(self, model, checkpoint_path, parts):
        # Load the entire checkpoint
        checkpoint_path = os.path.join(self.opt.log_dir, checkpoint_path)
        state_dict = torch.load(checkpoint_path)

        # Filter out unwanted keys
        filtered_state_dict = {k: v for k, v in state_dict.items() if any(part in k for part in parts)}

        # Update the model's state_dict
        model.load_state_dict(filtered_state_dict, strict=False)
        print('Loaded pre_trained {} of {}'.format(os.path.basename(checkpoint_path), parts))

        return model

    def fix_model_state_dict(self, state_dict):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k
            if name.startswith('module.'):
                name = name[7:]  # remove 'module.' of dataparallel
            new_state_dict[name] = v
        return new_state_dict