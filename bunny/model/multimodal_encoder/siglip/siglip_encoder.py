import torch
import torch.nn as nn

from transformers import SiglipVisionModel, SiglipImageProcessor, SiglipVisionConfig
from bunny.util.s2wrapper import forward as multiscale_forward
from bunny.util.merge import bipartite_soft_matching_merge

# SiglipVisionTower 模块的功能和之前我们讨论的 CLIPVisionTower 基本类似，都是封装了一个预训练的视觉Transformer模型来做视觉特征提取。核心功能是：

# 从路径或名称加载预训练的SigLip视觉模型和对应的图片预处理器。

# 冻结模型参数，不参与训练，只进行推理。

# 通过指定的层(select_layer)和特征类型（patch或包含CLS token的cls_patch）选取需要的隐藏特征。

# 支持输入单张图像或图像列表，输出对应的视觉特征。

# 附带一些属性方便获得隐藏向量维度、设备、数据类型等信息。

# 和 CLIPVisionTower 不同的地方可能是模型具体是SigLip架构，而不是CLIP架构，这意味着视觉编码器是基于SigLip设计的。这两个视觉模型的结构细节和训练目标或预训练任务有所区别，但在多模态架构中作为视觉编码器的用途和接口设计基本类似。

# 所以总结：

# SiglipVisionTower 是用来获得SigLip视觉模型的表示，作为多模态学习中的视觉特征提取部分。

# 代码实现和CLIP视觉编码器封装类似，封装预训练模型，冻结权重，提供接口提取指定层的视觉token特征。

# 具体区别在于底层视觉编码器模型不同，SigLip是一个新兴的视觉Transformer模型。


class SiglipVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = -2

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = SiglipVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self):
        if self.is_loaded:
            return
        self.image_processor = SiglipImageProcessor.from_pretrained(self.vision_tower_name)
        self.image_processor.crop_size = self.image_processor.size
        self.vision_tower = SiglipVisionModel.from_pretrained(self.vision_tower_name)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]

        return image_features


    #这里有可能是一个隐藏的bug，如果要处理的是多个影像的时候，如何返回中间层结果
    def forward(self, images):
        
        if type(images) is list:
            image_features = []
            image_forward_outs = [] 
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                                                      output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
                image_forward_outs.append(image_forward_out)
            image_forward_outs = torch.cat(image_forward_outs, dim=0)

        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype),
                                                   output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)
            

        return image_features, image_forward_outs

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2


class SiglipVisionTowerS2(SiglipVisionTower):
    def __init__(self, vision_tower, args, delay_load=False):
        self.s2_scales = getattr(args, 's2_scales', '384,768,1152')
        self.s2_scales = list(map(int, self.s2_scales.split(',')))
        self.s2_scales.sort()
        self.s2_split_size = self.s2_scales[0]
        self.s2_image_size = self.s2_scales[-1]

        super().__init__(vision_tower, args, delay_load)

        self.multiscale_forward = multiscale_forward

        if not delay_load:
            self.image_processor.size['height'] = self.image_processor.size['width'] = self.s2_image_size
            self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

    def load_model(self):
        if self.is_loaded:
            return
        self.image_processor = SiglipImageProcessor.from_pretrained(self.vision_tower_name)
        self.image_processor.crop_size = self.image_processor.size
        self.vision_tower = SiglipVisionModel.from_pretrained(self.vision_tower_name)
        self.vision_tower.requires_grad_(False)

        self.image_processor.size['height'] = self.image_processor.size['width'] = self.s2_image_size
        self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

        self.is_loaded = True

    def forward_feature(self, images):
        image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype),
                                               output_hidden_states=True)
        image_features = self.feature_select(image_forward_outs).to(images.dtype)
        return image_features

    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_feature = self.multiscale_forward(self.forward_feature, image.unsqueeze(0),
                                                        img_sizes=self.s2_scales, max_split_size=self.s2_split_size)
                

                #r = image_feature.shape[1] // 2
                #image_feature = bipartite_soft_matching_merge(image_feature,r,image_feature)
                image_features.append(image_feature)
        else:
            image_features = self.multiscale_forward(self.forward_feature, images, img_sizes=self.s2_scales,
                                                     max_split_size=self.s2_split_size)
            
            #r = image_features.shape[1] // 2
            #image_features = bipartite_soft_matching_merge(image_features,r,image_features)


        return image_features

    @property
    def hidden_size(self):
        return self.config.hidden_size * len(self.s2_scales)
    

    @property
    def patch_size(self):
        return self.config.patch_size
