import torch
import torch.nn as nn

from .eva_clip_processors import EvaClipImageTrainProcessor
from .eva_vit import Eva2LargePlusEncoder



#对影像数据进行与预先处理
class EvaClipVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_path = vision_tower
        self.config = VisionTowerConfig()

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = self.config


# image单张图像的shape一般是 (channels, height, width)，unsqueeze(0)后变成 (1, channels, height, width)，表示batch size = 1。

# self.vision_tower返回的image_feature一般是 (1, num_patches, hidden_size)。

# 由于你逐张处理并把每个image_feature放入列表，所以:

# image_features 是一个list，里面每个元素的形状是(1, num_patches, hidden_size)。

# 如果你想合并成单个tensor，可以用torch.cat(image_features, dim=0)，最终shape变为 (batch_size, num_patches, hidden_size)。


    def load_model(self):
        if self.is_loaded:
            return
        self.image_processor = EvaClipImageTrainProcessor(self.config.image_size)
        self.vision_tower = Eva2LargePlusEncoder(self.vision_tower_path)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_feature = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0)).to(
                    image.dtype)
                image_features.append(image_feature)
        else:
            image_features = self.vision_tower(images.to(device=self.device, dtype=self.dtype)).to(images.dtype)

        return image_features

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2


class VisionTowerConfig():
    def __init__(self):
        self.image_size = 336
        self.patch_size = 14
        self.hidden_size = 1024
