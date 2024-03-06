import torch
import torch.nn as nn
from torchvision import models
import math
from .network_utils import *
from utils.util import get_kinematic_parents

######################################################################################
# Network structure
######################################################################################

def get_limb_dim(opt):
    if opt.heatmap_type == 'none':
        limb_heatmap_dim = 0
    elif opt.heatmap_type == 'sin':
        limb_heatmap_dim = 2
    elif opt.heatmap_type == 'limb':
        limb_heatmap_dim = 1
        
    return limb_heatmap_dim


############################## UnrealEgo ##############################

class HeatMap_UnrealEgo_Shared(nn.Module):
    def __init__(self, opt, model_name='resnet18', input_channel_scale=2):
        super(HeatMap_UnrealEgo_Shared, self).__init__()

        self.backbone = HeatMap_UnrealEgo_Shared_Backbone(opt, model_name=model_name)
        self.after_backbone = HeatMap_UnrealEgo_AfterBackbone(opt, model_name=model_name, input_channel_scale=input_channel_scale)

    def forward(self, *inputs):
        backbone_outputs = self.backbone(*inputs)
        output = self.after_backbone(*backbone_outputs)

        return output

class HeatMap_UnrealEgo_Shared_Backbone(nn.Module):
    def __init__(self, opt, model_name='resnet18'):
        super(HeatMap_UnrealEgo_Shared_Backbone, self).__init__()
        self.joint_preset = opt.joint_preset
            
        self.backbone = Encoder_Block(opt, model_name=model_name)

    def forward(self, *inputs):
        outputs = []
        for input in inputs:
            output = self.backbone(input)
            outputs.append(output)
        outputs = tuple(outputs)
        return outputs

class Encoder_Block(nn.Module):
    def __init__(self, opt, model_name='resnet18'):
        super(Encoder_Block, self).__init__()

        if model_name == 'resnet18':
            self.backbone = models.resnet18(pretrained=opt.init_ImageNet)
        elif model_name == "resnet34":
            self.backbone = models.resnet34(pretrained=opt.init_ImageNet)
        elif model_name == "resnet50":
            self.backbone = models.resnet50(pretrained=opt.init_ImageNet)
        elif model_name == "resnet101":
            self.backbone = models.resnet101(pretrained=opt.init_ImageNet)
        else:
            raise NotImplementedError('model type [%s] is invalid', model_name)

        self.base_layers = list(self.backbone.children())
        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)

    def forward(self, input):
        
        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        output = [input, layer0, layer1, layer2, layer3, layer4]

        return output

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=2):
        super(ResidualBlock, self).__init__()
        self.conv1 = convrelu(in_channels, out_channels, kernel_size, padding)
        self.conv2 = convrelu(out_channels, out_channels, kernel_size, padding)
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += residual
        return out
    
class HeatMap_UnrealEgo_AfterBackbone(nn.Module):
    def __init__(self, opt, model_name="resnet18", input_channel_scale=2):
        super(HeatMap_UnrealEgo_AfterBackbone, self).__init__()

        if model_name == 'resnet18':
            feature_scale = 1
        elif model_name == "resnet34":
            feature_scale = 1
        elif model_name == "resnet50":
            feature_scale = 4
        elif model_name == "resnet101":
            feature_scale = 4
        else:
            raise NotImplementedError('model type [%s] is invalid', model_name)

        feature_scale *= input_channel_scale

        limb_heatmap_dim = get_limb_dim(opt)
        self.num_heatmap = opt.num_heatmap + opt.num_rot_heatmap * limb_heatmap_dim

        # self.layer0_1x1 = convrelu(128, 128, 1, 0)
        self.layer1_1x1 = convrelu(64 * feature_scale, 64 * feature_scale, 1, 0)
        self.layer2_1x1 = convrelu(128 * feature_scale, 128 * feature_scale, 1, 0)
        self.layer3_1x1 = convrelu(256 * feature_scale, 258 * feature_scale, 1, 0)
        self.layer4_1x1 = convrelu(512 * feature_scale, 512 * feature_scale, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        conv_up3_in_ch = 258 * feature_scale + 512 * feature_scale
        conv_up2_in_ch = 128 * feature_scale + 512 * feature_scale
        conv_up1_in_ch = 64 * feature_scale + 256 * feature_scale

        self.conv_up3 = convrelu(conv_up3_in_ch, 512 * feature_scale, 3, 1)
        self.conv_up2 = convrelu(conv_up2_in_ch, 256 * feature_scale, 3, 1)
        self.conv_up1 = convrelu(conv_up1_in_ch, 256 * feature_scale, 3, 1)

        self.conv_heatmap = nn.Conv2d(256 * feature_scale, self.num_heatmap * input_channel_scale, 1)


    def forward(self, *list_inputs):
        list_features = []
        for id in range(len(list_inputs[0])):
            id_inputs_list = [list_inputs[i][id] for i in range(len(list_inputs))]
            list_features.append(torch.cat(id_inputs_list, dim=1))
                
        input = list_features[0] # size = [16, 6, 256, 256]
        layer0 = list_features[1] # size = [16, 128, 128, 128]
        layer1 = list_features[2] # size = [16, 128, 64, 64]
        layer2 = list_features[3] # size = [16, 256, 32, 32]
        layer3 = list_features[4] # size = [16, 512, 16, 16]
        layer4 = list_features[5] # size = [16, 1024, 8, 8]

        layer4 = self.layer4_1x1(layer4) # size = [16, 1024, 8, 8]
        x = self.upsample(layer4) # size = [16, 1024, 16, 16]
        layer3 = self.layer3_1x1(layer3) # size = [16, 516, 16, 16]
        
        x = torch.cat([x, layer3], dim=1) # size = [16, 1540, 16, 16]
        x = self.conv_up3(x) # size = [16, 1024, 16, 16]

        x = self.upsample(x) # size = [16, 1024, 32, 32]
        layer2 = self.layer2_1x1(layer2) # size = [16, 256, 32, 32]
        
        x = torch.cat([x, layer2], dim=1) # size = [16, 1280, 32, 32]
        x = self.conv_up2(x) # size = [16, 512, 32, 32]

        x = self.upsample(x) # size = [16, 512, 64, 64]
        layer1 = self.layer1_1x1(layer1) # size = [16, 128, 64, 64]
        
        x = torch.cat([x, layer1], dim=1) # size = [16, 640, 64, 64]
        x = self.conv_up1(x) # size = [16, 512, 64, 64]

        output = self.conv_heatmap(x) # size = [16, 30, 64, 64]

        return output


############################## AutoEncoder ##############################


class MLPDecoder(nn.Module):
    def __init__(self, opt, input_dim, output_dim, fc_layers=None):
        ## pose decoder
        super(MLPDecoder, self).__init__()
        fc_layers = [32, 32] if fc_layers is None else fc_layers
        
        self.with_bn = True
        self.with_pose_relu = True
        self.pose_fcs = []
        layer_dims = [input_dim] + fc_layers
        
        if len(layer_dims) != 3:
            for i, (in_dim, out_dim) in enumerate(zip(layer_dims[:-1], layer_dims[1:])):
                self.pose_fcs.append(make_fc_layer(in_dim, out_dim, with_relu=self.with_pose_relu, with_bn=self.with_bn))
            self.pose_fcs.append(torch.nn.Linear(layer_dims[-1], output_dim))
            self.pose_fcs = nn.ModuleList(self.pose_fcs)
        elif len(layer_dims) == 3:
            # Code for backward compatibility
            self.pose_fc1 = make_fc_layer(layer_dims[0], layer_dims[1], with_relu=self.with_pose_relu, with_bn=self.with_bn)
            self.pose_fc2 = make_fc_layer(layer_dims[1], layer_dims[2], with_relu=self.with_pose_relu, with_bn=self.with_bn)
            self.pose_fc3 = torch.nn.Linear(layer_dims[2], output_dim)
            self.pose_fcs = [self.pose_fc1, self.pose_fc2, self.pose_fc3]
        
    def forward(self, input):
        if len(self.pose_fcs) > 0 and len(self.pose_fcs) != 3:
            for fc in self.pose_fcs:
                input = fc(input)
            return input
        elif len(self.pose_fcs) == 3:
            input = self.pose_fc1(input)
            input = self.pose_fc2(input)
            return self.pose_fc3(input)
        else:
            raise ValueError("Invalid MLPDecoder")
        

class Encoder_Block(nn.Module):
    def __init__(self, opt, model_name='resnet18'):
        super(Encoder_Block, self).__init__()

        if model_name == 'resnet18':
            self.backbone = models.resnet18(pretrained=opt.init_ImageNet)
        elif model_name == "resnet34":
            self.backbone = models.resnet34(pretrained=opt.init_ImageNet)
        elif model_name == "resnet50":
            self.backbone = models.resnet50(pretrained=opt.init_ImageNet)
        elif model_name == "resnet101":
            self.backbone = models.resnet101(pretrained=opt.init_ImageNet)
        else:
            raise NotImplementedError('model type [%s] is invalid', model_name)

        self.base_layers = list(self.backbone.children())
        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)

    def forward(self, input):
        
        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        output = [input, layer0, layer1, layer2, layer3, layer4]

        return output
    
class HeatmapFeatureExtractorFC(nn.Module):
    def __init__(self, opt, num_heatmap=30, hidden_size=10, with_bn=True, channels=1):
        super().__init__()
        self.with_bn = with_bn
        self.hidden_size = hidden_size
        self.num_heatmap = num_heatmap
        self.channels = channels
        
        self.hm_size = opt.load_size_heatmap
        
        self.fc1 = make_fc_layer(in_feature=self.hm_size[0] * self.hm_size[1] * self.channels, out_feature=2048, with_bn=self.with_bn)
        self.fc2 = make_fc_layer(in_feature=2048, out_feature=512, with_bn=self.with_bn)
        self.fc3 = make_fc_layer(in_feature=512, out_feature=self.hidden_size, with_bn=self.with_bn)
        
    def forward(self, input):
        batch_size = input.size(0)
        
        assert input.size(1) == self.num_heatmap // self.channels
        input = input.reshape(batch_size * self.num_heatmap // self.channels, self.channels * self.hm_size[0] * self.hm_size[1])
        
        z = self.fc1(input)
        z = self.fc2(z)
        z = self.fc3(z) 
        
        z = z.reshape(batch_size, -1)
        return z
        
class HeatmapFeatureExtractor(nn.Module):
    def __init__(self, opt, num_heatmap=30, hidden_size=10, with_bn=True, channels=1):
        super(HeatmapFeatureExtractor, self).__init__()
        self.with_bn = with_bn
        self.hidden_size = hidden_size
        self.num_heatmap = num_heatmap
        self.channels = channels
        
        conv_ch = [num_heatmap, 32, 64, 128]
        conv_input_ch = conv_ch[:-1].copy()
        conv_output_ch = conv_ch[1:].copy()
        
        self.hm_size = opt.load_size_heatmap
        
        self.fc_dim = (self.hm_size[0] // 8) * (self.hm_size[1] // 8) * conv_ch[-1]
        self.conv1 = make_conv_layer(in_channels=conv_input_ch[0], out_channels=conv_output_ch[0], kernel_size=4, stride=2, padding=1, with_bn=self.with_bn)
        self.conv2 = make_conv_layer(in_channels=conv_input_ch[1], out_channels=conv_output_ch[1], kernel_size=4, stride=2, padding=1, with_bn=self.with_bn)
        self.conv3 = make_conv_layer(in_channels=conv_input_ch[2], out_channels=conv_output_ch[2], kernel_size=4, stride=2, padding=1, with_bn=self.with_bn)

        self.fc1 = make_fc_layer(in_feature=self.fc_dim, out_feature=2048, with_bn=self.with_bn)
        self.fc2 = make_fc_layer(in_feature=2048, out_feature=512, with_bn=self.with_bn)
        self.fc3 = make_fc_layer(in_feature=512, out_feature=self.hidden_size, with_bn=self.with_bn)
    
    def forward(self, input):
        batch_size = input.size(0)
        
        if self.is_indep:
            assert input.size(1) == self.num_heatmap // self.channels
            input = input.reshape(batch_size * self.num_heatmap // self.channels, self.channels, self.hm_size[0], self.hm_size[1])
        
        # encode heatmap
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)

        x = x.view(batch_size, -1)
            
        x = self.fc1(x)
        x = self.fc2(x)
        z = self.fc3(x)
        
        return z
    
from .modeling_vit import ViTModel, ViTConfig
class PatchedHeatmapFeatureExtractorViT(nn.Module):
    def __init__(self, opt, num_heatmap=30, hidden_size=10, num_channels=1, num_layers=3):
        super(PatchedHeatmapFeatureExtractorViT, self).__init__()
        hm_size = opt.load_size_heatmap
        self.num_heatmap = num_heatmap
        self.heatmap_size =  hm_size[0]
        self.patch_size = 16
        assert hm_size[0] == hm_size[1] and hm_size[0] % self.patch_size == 0
        self.image_size = (int(math.sqrt(num_heatmap-1)) + 1) * self.heatmap_size
        self.num_dummies = (self.image_size ** 2 - self.num_heatmap * self.heatmap_size ** 2) // (self.heatmap_size ** 2)
        
        self.image_heatmap_divisor = self.image_size // self.heatmap_size
        self.heatmap_patch_divisor = self.heatmap_size // self.patch_size
        self.num_patches_per_heatmap = self.heatmap_patch_divisor ** 2
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.hm_channels = num_channels
        self.num_layers = num_layers
        
        if self.num_layers > 0:
            # Using the ViT configuration to define our ViT structure.
            self.vit_hidden_size = 1024
            config = ViTConfig(
                image_size=self.image_size, # Image size
                patch_size=self.patch_size, # Size of each patch
                num_channels=self.hm_channels,   # Channels of the heatmap
                num_hidden_layers=self.num_layers,       # Number of transformer layers
                num_attention_heads=8,
                intermediate_size=self.vit_hidden_size * 4,
                hidden_size=self.vit_hidden_size,
                # position_embedding_type='absolute'
            )
            
            self.dummy_mask = torch.zeros((self.image_heatmap_divisor * self.image_heatmap_divisor), dtype=torch.bool)
            self.dummy_mask[num_heatmap:] = True
            self.dummy_mask = self.dummy_mask.reshape(self.image_heatmap_divisor, self.image_heatmap_divisor)
            self.dummy_mask = self.dummy_mask.kron(torch.ones((self.heatmap_patch_divisor, self.heatmap_patch_divisor), dtype=torch.bool))
            self.dummy_mask = self.dummy_mask.reshape(-1)
            
            self.vit = ViTModel(config, use_mask_token=True, use_cls_token=False)
            
            # Following the structure from your model to process the output from the ViT
            vit_hidden_size = config.hidden_size * (self.heatmap_size // self.patch_size) ** 2
        else:
            vit_hidden_size = hm_size[0] * hm_size[1] * num_channels
        
        # self.fc3 = make_fc_layer(in_feature=vit_hidden_size, out_feature=hidden_size, with_bn=True)
        self.fc1 = make_fc_layer(in_feature=vit_hidden_size, out_feature=2048, with_bn=True)
        self.fc2 = make_fc_layer(in_feature=2048, out_feature=512, with_bn=True)
        self.fc3 = make_fc_layer(in_feature=512, out_feature=hidden_size, with_bn=True)
    
    def forward(self, input):
        batch_size = input.size(0)
        
        assert input.size(1) == self.num_heatmap \
            and input.size(2) == self.hm_channels
        input = input.transpose(1, 2)
        input = input.reshape(batch_size, self.hm_channels, self.num_heatmap, self.heatmap_size, self.heatmap_size)
        
        # Encode heatmap through ViT
        dummy_patches = torch.zeros((batch_size, self.hm_channels, self.num_dummies,
                                     self.heatmap_size, self.heatmap_size), dtype=torch.float32, device=input.device)
        input = torch.cat([input, dummy_patches], dim=2)
        input = input.reshape(batch_size, self.hm_channels, self.image_heatmap_divisor, self.image_heatmap_divisor, self.heatmap_size, self.heatmap_size)
        input = input.permute(0, 1, 2, 4, 3, 5).reshape(batch_size, self.hm_channels, self.image_size, self.image_size)
        
        outputs = self.vit(pixel_values=input, bool_masked_pos=self.dummy_mask.to(input.device))
        
        all_patch_embeddings = outputs.last_hidden_state
        all_patch_embeddings = all_patch_embeddings.reshape(batch_size,
                                                            (self.image_size // self.patch_size), 
                                                            (self.image_size // self.patch_size),
                                                            self.vit_hidden_size)
        per_heatmap_embeddings = torch.zeros((batch_size,
                                              self.num_heatmap,
                                              self.heatmap_patch_divisor * self.heatmap_patch_divisor * all_patch_embeddings.size(-1)
                                              ), dtype=torch.float32, device=input.device)
        
        for i in range(self.num_heatmap):
            col = i % self.image_heatmap_divisor
            row = i // self.image_heatmap_divisor
            col *= self.heatmap_patch_divisor
            row *= self.heatmap_patch_divisor
            per_heatmap_embeddings[:, i, :] = all_patch_embeddings[:, row:row+self.heatmap_patch_divisor, col:col+self.heatmap_patch_divisor, :].reshape(batch_size, -1)

        x = per_heatmap_embeddings

        x = x.view(batch_size * self.num_heatmap, -1)
        
        # Continue with fully connected layers
        x = self.fc1(x)
        x = self.fc2(x)
        z = self.fc3(x)
        
        z = z.reshape(batch_size, -1)
        
        return z
    
class HeatmapDecoder(nn.Module):
    def __init__(self, opt, num_heatmap=30, fc_dim=16384, hidden_size=10, with_bn=True, is_indep=False, channels=1):
        super(HeatmapDecoder, self).__init__()
        self.num_heatmap = num_heatmap
        self.hidden_size = hidden_size
        
        hm_size = opt.load_size_heatmap
        self.fc_dim = (hm_size[0] // 8) * (hm_size[0] // 8) * 128
        self.with_bn = with_bn
        self.is_indep = is_indep
        self.channels = channels
        
        # heatmap decoder
        self.heatmap_fc1 = make_fc_layer(self.hidden_size, 512, with_bn=self.with_bn)
        self.heatmap_fc2 = make_fc_layer(512, 2048, with_bn=self.with_bn)
        self.heatmap_fc3 = make_fc_layer(2048, self.fc_dim, with_bn=self.with_bn)
        self.W = opt.load_size_heatmap[0]
        self.H = opt.load_size_heatmap[1]

        self.deconv1 = make_deconv_layer(128, 64, kernel_size=4, stride=2, padding=1, with_bn=self.with_bn)
        self.deconv2 = make_deconv_layer(64, 32, kernel_size=4, stride=2, padding=1, with_bn=self.with_bn)
        self.deconv3 = make_deconv_layer(32, self.num_heatmap if not is_indep else channels, kernel_size=4, stride=2, padding=1, with_bn=self.with_bn)

    def forward(self, input):
        z = input
        batch_size = z.size(0)
        
        if self.is_indep:
            z = z.reshape(batch_size * self.num_heatmap // self.channels, -1)
        
        # decode heatmap
        x_hm = self.heatmap_fc1(z)
        x_hm = self.heatmap_fc2(x_hm)
        x_hm = self.heatmap_fc3(x_hm) 
        
        if not self.is_indep:
            x_hm = x_hm.view(batch_size, 128, self.W // 8, self.H // 8)
        else:
            x_hm = x_hm.view(batch_size * self.num_heatmap // self.channels, 128, self.W // 8, self.H // 8)
        
        x_hm = self.deconv1(x_hm)
        x_hm = self.deconv2(x_hm)
        output_hm = self.deconv3(x_hm)
        
        if self.is_indep:
            output_hm = output_hm.reshape(batch_size, self.num_heatmap, self.W, self.H)
        
        return output_hm

class SkelNet(nn.Module):
    def __init__(self, opt, input_size, bridge_size, num_layers, batch_first=False, layer_type="LSTM"):
        super(SkelNet, self).__init__()
        self.batch_first = batch_first
        self.kinematic_parents = get_kinematic_parents(opt.joint_preset)
        self.n_root_joint = 1
        self.n_nodes = len(self.kinematic_parents) - self.n_root_joint
        
        self.input_size = input_size
        self.bridge_size = bridge_size
        self.output_size = input_size + bridge_size
        self.num_layers = num_layers
        
        self.mode = layer_type
        
        if self.mode == "LSTMSplit" or self.mode == "LSTMNoRel" or self.mode == "NoneNoRel":
            assert input_size == bridge_size
            self.output_size = input_size
            
        self.split_bridge = self.mode == "PU"

        self.lstm = None
        if self.mode == "LSTM" or self.mode == "LSTMSplit" or self.mode == "LSTMNoRel":
            self.lstm = make_lstm_layer(in_feature=self.output_size,
                                        hidden_size=self.output_size,
                                        num_layers=self.num_layers,
                                        batch_first=True)

        elif self.mode == "PU":
            self.lstm_custom = make_pu_layer(in_feature=self.output_size // 2,
                                       bridge_size=self.output_size // 2,
                                        hidden_size=self.output_size,
                                         num_layers=self.num_layers,
                                         batch_first=True)

        elif self.mode == "None" or self.mode == "NoneNoRel":
            pass
        else:
            raise ValueError("Invalid SkelNet layer type")
            
    def get_output_size(self):
        return self.output_size
    
    def flatten_parameters(self):
        if self.lstm is not None:
            self.lstm.flatten_parameters()
        
    def forward(self, input, bridge):
        self.flatten_parameters()
        
        if not self.batch_first:
            input = input.transpose(0, 1)
            bridge = bridge.transpose(0, 1)
        
        all_inputs = torch.cat([input, bridge], dim=-1)
        if self.mode in ["LSTM", "None"]:
            input = all_inputs
            bridge = torch.zeros(input.size(0), input.size(1), 0).to(input.device)
        
        if self.mode == "None" or self.mode == "NoneNoRel":
            if not self.batch_first:
                input = input.transpose(0, 1)
            return input

        batch_size = input.size(0)
        hs, cs = [], []
        
        for i in range(self.n_root_joint):
            hs.append(torch.zeros(self.num_layers, batch_size, self.output_size).to(input.device))
            cs.append(torch.zeros(self.num_layers, batch_size, self.output_size).to(input.device))
            
        hier_output = []
        
        for i in range(self.n_root_joint, len(self.kinematic_parents)):
            parent_idx = self.kinematic_parents[i]
            parent_h = hs[parent_idx]
            parent_c = cs[parent_idx]
            
            if self.mode == "LSTM":
                joint_output, hc = self.lstm(input[:, [i - self.n_root_joint]], (parent_h, parent_c))
                
            elif self.mode == "LSTMSplit":
                joint_output, hc = self.lstm(bridge[:, [i - self.n_root_joint]], (parent_h, parent_c))
                joint_output, hc = self.lstm(input[:, [i - self.n_root_joint]], (hc[0], hc[1]))
                
            elif self.mode == "LSTMNoRel":
                joint_output, hc = self.lstm(input[:, [i - self.n_root_joint]], (parent_h, parent_c))
                
            elif self.mode == "PU":
                joint_output, hc = self.lstm_custom(input[:, [i - self.n_root_joint]], bridge[:, [i - self.n_root_joint]], (parent_h, parent_c))
                
            elif self.mode == "FC" or \
                self.mode == "FCNoRel":
                break
            else:
                joint_output, hc = self.lstm(input[:, [i - self.n_root_joint]], parent_h)
                hc = (hc, hc)
            
            joint_output = joint_output[:, 0]
            
            hier_output.append(joint_output)
            hs.append(hc[0])
            cs.append(hc[1])
            
        else:
            hier_output = torch.stack(hier_output, dim=1)
        
        if not self.batch_first:
            hier_output = hier_output.transpose(0, 1)

        return hier_output


class EgoTAPAutoEncoder(nn.Module):
    def __init__(self, opt, input_channel_scale=1, fc_dim=16384):
        super(EgoTAPAutoEncoder, self).__init__()
        
        self.hierarchy_transformer = False

        self.joint_preset = opt.joint_preset
        self.hidden_size = opt.ae_hidden_size
        
        self.limb_heatmap_dim = get_limb_dim(opt)
        self.with_bn = True
        self.with_pose_relu = True
        
        self.num_joints = opt.num_heatmap
        if opt.estimate_head:
            self.num_joints += 1

        self.num_pos_heatmap = opt.num_heatmap
        self.num_rot_heatmap = opt.num_rot_heatmap
        assert self.num_pos_heatmap == self.num_rot_heatmap
        self.n_encode_joints = self.num_pos_heatmap
        
        self.use_rot_heatmap = opt.patched_heatmap_ae
        self.num_heatmap = self.num_pos_heatmap
        if self.use_rot_heatmap:
            self.num_heatmap += self.num_rot_heatmap * self.limb_heatmap_dim

        self.input_channel_scale = input_channel_scale
        self.channels_heatmap = self.num_heatmap * input_channel_scale
        
        self.W = opt.load_size_heatmap[0]
        self.H = opt.load_size_heatmap[1]
        
        self.pose_dim = self.num_joints * 3
        
        self.joint_preset = opt.joint_preset
        self.kinmatic_parents = get_kinematic_parents(self.joint_preset)
            
        self.rot_dim = self.num_rot_heatmap * 3
        
        self.use_patched_heatmap_ae = opt.patched_heatmap_ae
        
        pose_input_dim = self.hidden_size
        self.body_hidden_size = self.hidden_size
        self.body_hidden_size *= input_channel_scale
        
        self.use_global_offset = opt.joint_preset == "UnrealEgo" and opt.estimate_head
        
        if opt.patched_heatmap_ae:
            if self.num_pos_heatmap > 0:
                self.pos_heatmap_encoder = PatchedHeatmapFeatureExtractorViT(opt, self.num_pos_heatmap * self.input_channel_scale,
                                                                            self.hidden_size)

            if self.num_rot_heatmap > 0:
                self.rot_heatmap_encoder = HeatmapFeatureExtractorFC(opt, self.num_rot_heatmap * self.limb_heatmap_dim * self.input_channel_scale,
                                                    self.hidden_size, channels=self.limb_heatmap_dim)
            pose_input_dim *= self.num_pos_heatmap + self.num_rot_heatmap
            pose_input_dim *= self.input_channel_scale
        else:
            self.heatmap_encoder = HeatmapFeatureExtractor(opt, self.channels_heatmap, self.hidden_size)
        
        self.feature_size = self.hidden_size
        
        if self.use_patched_heatmap_ae:
            # Propagation layers replaces mlp layers
            pose_mlp_dim = []
            
            self.skel_layer_type = opt.skel_layer
            num_skel_layers = 2
            if hasattr(opt, 'n_skel_layers'):
                num_skel_layers = opt.n_skel_layers
            self.skel_sequential_layer = SkelNet(opt, input_size=self.body_hidden_size, bridge_size=self.body_hidden_size,
                                                 num_layers=num_skel_layers, batch_first=False, layer_type=self.skel_layer_type)
            embedding_size = self.skel_sequential_layer.get_output_size()
            
            pose_input_dim += embedding_size * self.n_encode_joints - self.body_hidden_size * self.input_channel_scale * self.n_encode_joints
            
            self.feature_size = embedding_size
            
        else:
            embedding_size = opt.ae_hidden_size
            pose_mlp_dim = [embedding_size, embedding_size]

        self.indep_projection = (not (opt.skel_layer == "FC" or opt.skel_layer == "FCNoRel")) and self.use_patched_heatmap_ae
        if self.indep_projection:
            assert self.use_patched_heatmap_ae
            self.indep_decode_size = self.feature_size
            self.indep_decode_size += self.body_hidden_size

            self.pose_mlp = MLPDecoder(opt, self.indep_decode_size, 3, fc_layers=pose_mlp_dim)
            
            self.global_pose_dim = 3 * (self.num_joints - self.num_pos_heatmap)
            if self.use_global_offset:
                self.global_pose_dim += 3
            if self.global_pose_dim > 0:
                self.global_mlp = MLPDecoder(opt, pose_input_dim, self.global_pose_dim, fc_layers=pose_mlp_dim)
        else:
            self.pose_mlp = MLPDecoder(opt, pose_input_dim, self.pose_dim, fc_layers=pose_mlp_dim)


    def predict_pose(self, input, input_rgb_left=None, input_rgb_right=None):
        return self.forward(input, input_rgb_left, input_rgb_right, pose_only=True)

    def forward(self, input, input_rgb_left=None, input_rgb_right=None, pose_only=False):
        with torch.set_grad_enabled(True):
            batch_size = input.size()[0]
            
            n_encode_joints = self.num_pos_heatmap # Assume the same number of pos, rot heatmaps
            if self.use_patched_heatmap_ae:
                pos_input = input[:, :self.num_pos_heatmap * self.input_channel_scale]
                pos_input = pos_input.reshape(batch_size, self.num_pos_heatmap * self.input_channel_scale, 1, input.size(-2), input.size(-1))
                rot_input = input[:, self.num_pos_heatmap * self.input_channel_scale:]
                rot_input = rot_input.reshape(batch_size, self.input_channel_scale, self.limb_heatmap_dim, self.num_rot_heatmap, input.size(-2), input.size(-1))
                rot_input = rot_input.swapaxes(1, 2)
                rot_input = rot_input.reshape(batch_size, self.limb_heatmap_dim, self.input_channel_scale * self.num_rot_heatmap, input.size(-2), input.size(-1))
                rot_input = rot_input.swapaxes(1, 2)
                
                pos_embed = self.pos_heatmap_encoder(pos_input)
                rot_embed = self.rot_heatmap_encoder(rot_input)
                
                pos_embed = pos_embed.reshape(batch_size, self.input_channel_scale, self.num_pos_heatmap, self.hidden_size)
                pos_embed = pos_embed.swapaxes(1, 2)
                pos_embed = pos_embed.reshape(batch_size, -1)
                
                rot_embed = rot_embed.reshape(batch_size, self.input_channel_scale, self.num_rot_heatmap, self.hidden_size)
                rot_embed = rot_embed.swapaxes(1, 2)
                rot_embed = rot_embed.reshape(batch_size, -1)
                
                z = torch.cat((pos_embed, rot_embed), dim=-1)
                z_bar = z
                
            else:
                if not self.use_rot_heatmap:
                    pos_input = input[:, :self.num_pos_heatmap * self.input_channel_scale]
                    z = self.heatmap_encoder(pos_input)
                else:
                    z = self.heatmap_encoder(input)
                z_bar = z

            rot = torch.zeros((batch_size, self.rot_dim), device=z.device)
            indep_pos = torch.zeros((batch_size, 3 * 2 * self.num_pos_heatmap), device=z.device)

            if self.use_patched_heatmap_ae:
                skel_input = pos_embed.reshape(batch_size, n_encode_joints, -1).swapaxes(0, 1)
                skel_bridge = rot_embed.reshape(batch_size, n_encode_joints, -1).swapaxes(0, 1)
                
                self.skel_inputs = torch.cat((skel_input, skel_bridge), dim=-1)
                skel_embed = self.skel_sequential_layer(input=skel_input, bridge=skel_bridge)
                self.skel_embed = skel_embed
        
                skel_embed = skel_embed.swapaxes(0, 1).reshape(batch_size, -1)
                z_bar = skel_embed
            
            if self.indep_projection:
                feature_embed = z_bar
                per_joint_embed = feature_embed.reshape(batch_size, self.num_pos_heatmap, self.feature_size)
                
                pos_per_joint = pos_embed.reshape(batch_size, self.num_pos_heatmap, self.body_hidden_size)
                per_joint_embed = torch.cat((pos_per_joint, per_joint_embed), dim=-1)
                
                output_pose = self.pose_mlp(per_joint_embed.reshape(-1, self.indep_decode_size)).reshape(batch_size, -1)
                
                if self.global_pose_dim > 0:
                    other_poses = self.global_mlp(z_bar)
                    if self.use_global_offset:
                        global_offset = other_poses[:, :3]
                        output_pose = (output_pose.reshape(batch_size, -1, 3) + global_offset[:, None, :]).reshape(batch_size, -1)
                        other_poses = other_poses[:, 3:]
                    output_pose = torch.cat((output_pose, other_poses), dim=1)
            else:
                output_pose = self.pose_mlp(z_bar)

            output_pose = output_pose[:, :].view(batch_size, self.num_joints, 3)
            
            if pose_only:
                return output_pose
        
        output_hm = torch.zeros((batch_size, self.channels_heatmap, self.W, self.H), device=z.device)

        return output_pose, rot, indep_pos, output_hm
    

if __name__ == "__main__":
    
    model = HeatMap_UnrealEgo_Shared(opt=None, model_name='resnet50')

    input = torch.rand(3, 3, 256, 256)
    outputs = model(input, input)
    pred_heatmap_left, pred_heatmap_right = torch.chunk(outputs, 2, dim=1)

    print(pred_heatmap_left.size())
    print(pred_heatmap_right.size())
