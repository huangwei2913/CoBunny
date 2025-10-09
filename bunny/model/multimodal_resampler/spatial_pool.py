import torch
import torch.nn as nn
import math


# 这个SpatialPool模块是一个基于空间维度的特征池化(resampling)模块，用于视觉特征的下采样和通道数变换。

# 具体功能和作用如下：

# 输入的image_features是二维空间展开后的特征（形状是batch大小，token数，特征维度），根据输入的原始图像尺寸(images)，模块先还原特征的空间宽高维度(ori_W, ori_H)。

# 把特征reshape成二维空间结构 (B, C=特征维度F, H, W)。

# 根据构造参数mode选择不同池化方式：

# 'average'：使用二维平均池化(nn.AvgPool2d)。

# 'max'：使用二维最大池化(nn.MaxPool2d)。

# 'conv'：使用卷积(nn.Conv2d)进行空间降采样和通道变换。

# 池化核大小和步幅为stride，控制降采样比例。

# 最后将池化后二维特征展平回(batch, token数, 特征维度)格式输出，便于后续Transformer等模型处理。

# 总结：SpatialPool是一种视觉特征的空间维度池化器，能够对输入特征进行空间降采样（如缩小空间分辨率），减小后续模型处理计算量，同时支持通道数调整。通过平均池化、最大池化或者卷积实现灵活选择.


class SpatialPool(nn.Module):
    def __init__(self, model_args, vision_tower):
        super().__init__()

        self.mode = model_args.mm_spatial_pool_mode
        self.stride = model_args.mm_spatial_pool_stride
        # import pdb; pdb.set_trace()
        self.out_channels = getattr(model_args, 'mm_spatial_pool_out_channels', vision_tower.hidden_size)

        if self.mode == 'average':
            self.pool = nn.AvgPool2d(kernel_size=self.stride, stride=self.stride)
        elif self.mode == 'max':
            self.pool = nn.MaxPool2d(kernel_size=self.stride, stride=self.stride)
        elif self.mode == 'conv':
            self.pool = nn.Conv2d(in_channels=vision_tower.hidden_size, out_channels=self.out_channels, kernel_size=self.stride, stride=self.stride)
        else:
            raise ValueError(f'Unknown pooling mode: {self.pool}.')

    def forward(self, image_features, images, *args, **kwargs):
        ori_W = int(math.sqrt(image_features.shape[1] * images.shape[3] // images.shape[2]))
        ori_H = int(ori_W * images.shape[2] // images.shape[3])

        B, _, F = image_features.shape

        image_features_spatial = image_features.view(B, ori_H, ori_H, F).permute(0, 3, 1, 2)
        image_features_spatial_pool = self.pool(image_features_spatial)

        return image_features_spatial_pool.flatten(2).transpose(1, 2).contiguous()
    


    @property
    def out_channels(self):
        return self._out_channels    

    @property
    def config(self):
        return {
            'mm_resampler_type': 'spatial_pool',
            'mm_spatial_pool_stride': self.stride,
            'mm_spatial_pool_mode': self.mode,
            'mm_spatial_pool_out_channels': self.out_channels,
        }
