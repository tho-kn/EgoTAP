import torch
from torch.optim import lr_scheduler
from .net_architecture import *
from .network_utils import *


######################################################################################
# Define networks
######################################################################################

def define_HeatMap(opt, model):
    input_channel_scale = 2 if opt.stereo else 1
    if model == "egotap_autoencoder":
        net = HeatMap_UnrealEgo_Shared(opt, opt.model_name, input_channel_scale=input_channel_scale)
    elif model == "heatmap_shared":
        net = HeatMap_UnrealEgo_Shared(opt, opt.model_name, input_channel_scale=input_channel_scale)
    else:
        raise Exception("HeatMap is not implemented for {}".format(model))

    print_network_param(net, 'HeatMap_Estimator for {}'.format(model))

    return init_net(net, opt.init_type, opt.gpu_ids, opt.init_ImageNet)

def define_AutoEncoder(opt, model):
    input_channel_scale = 2 if opt.stereo else 1
    if model == "egotap_autoencoder":
        net = EgoTAPAutoEncoder(opt, input_channel_scale=input_channel_scale)
    else:
        raise Exception("AutoEncoder is not implemented for {}".format(model))

    print_network_param(net, 'AutoEncoder for {}'.format(model))

    return init_net(net, opt.init_type, opt.gpu_ids, False)

def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            # lr_l = 1.0 - max(0, epoch+1+1+opt.epoch_count-opt.niter) / float(opt.niter_decay+1)
            lr_l = 1.0 - max(0, epoch+opt.epoch_count-opt.niter) / float(opt.niter_decay+1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters_step, gamma=0.5)
    elif opt.lr_policy == 'exponent':
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    elif opt.lr_policy == 'cos_anneal':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=(opt.niter + opt.niter_decay) * opt.epoch_iter_cnt)
    # https://huggingface.co/docs/transformers/main_classes/optimizer_schedules#transformers.get_cosine_schedule_with_warmup.num_training_steps
    elif opt.lr_policy == 'cos_anneal_warmup':
        from transformers.optimization import get_cosine_schedule_with_warmup
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=opt.niter * opt.epoch_iter_cnt,
                                                    num_training_steps=(opt.niter + opt.niter_decay) * opt.epoch_iter_cnt)
    else:
        raise NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def get_optimizer(params, opt):
    if opt.optimizer_type == 'Adam':
        return torch.optim.Adam(
            params=params,
            lr=opt.lr,
            eps=opt.opt_eps,
            weight_decay=opt.weight_decay
        )
    elif opt.optimizer_type == 'SGD':
        return torch.optim.SGD(
            params=params,
            lr=opt.lr,
            weight_decay=opt.weight_decay
        )
    elif opt.optimizer_type == 'AdamW':
        return torch.optim.AdamW(
            params=params,
            lr=opt.lr,
            eps=opt.opt_eps,
            weight_decay=opt.weight_decay
        )
    elif opt.optimizer_type == 'DAdam':
        from dadaptation import DAdaptAdam
        return DAdaptAdam(
            params=params,
            lr=1.0,
            eps=opt.opt_eps,
            weight_decay=opt.weight_decay,
            growth_rate=opt.growth_rate,
            decouple=opt.decouple,
        )
    elif opt.optimizer_type == 'DSGD':
        from dadaptation import DAdaptSGD
        return DAdaptSGD(
            params=params,
            lr=1.0,
            weight_decay=opt.weight_decay,
            growth_rate=opt.growth_rate,
        )
    elif opt.optimizer_type == 'DAdaGrad':
        from dadaptation import DAdaptAdaGrad
        return DAdaptAdaGrad(
            params=params,
            lr=1.0,
            eps=opt.opt_eps,
            weight_decay=opt.weight_decay,
            growth_rate=opt.growth_rate,
        )
    elif opt.optimizer_type == 'Prodigy':
        from prodigyopt import Prodigy
        return Prodigy(
            params=params,
            lr=1.0,
            eps=opt.opt_eps,
            d_coef=opt.d_coef,
            growth_rate=opt.growth_rate,
            weight_decay=opt.weight_decay,
            safeguard_warmup=True,
        )
    else:
        raise NotImplementedError('optimizer type [%s] is not implemented', opt.optimizer_type)


def _freeze(*args):
    for module in args:
        if module:
            for p in module.parameters():
                p.requires_grad = False

def _unfreeze(*args):
    for module in args:
        if module:
            for p in module.parameters():
                p.requires_grad = True

def freeze_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()
        m.weight.requires_grad = False
        m.bias.requires_grad = False

def unfreeze_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.train()
        m.weight.requires_grad = True
        m.bias.requires_grad = True

def freeze_bn_affine(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.weight.requires_grad = False
        m.bias.requires_grad = False
