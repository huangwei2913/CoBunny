import torch
import torch.nn.functional as F
from modelscope import AutoImageProcessor, AutoModel, AutoConfig

from .base_encoder import BaseVisionTower
from bunny.util.merge import bipartite_soft_matching_merge

class DinoVisionTower(BaseVisionTower):
    def __init__(self, vision_tower, args, delay_load=False):
        super(DinoVisionTower, self).__init__(vision_tower, args, delay_load)
        self._vision_tower_name = vision_tower
        self._image_size = 224   #取一个基本默认值，
        self._patch_size = 16    #这个不代表最后的隐藏层获得的tokens数量
        self._num_patches_cached = None  # 缓存动态计算的patch数
        self.select_feature = 'cls_patch'
        if not self.delay_load:
            self.load_model()
        else:
            self.cfg_only = AutoConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        self.vision_tower = AutoModel.from_pretrained(self._vision_tower_name)
        self.image_processor = AutoImageProcessor.from_pretrained(self._vision_tower_name)
        self._hidden_size = self.vision_tower.layer_norm.normalized_shape[0]
        self._image_size = self.vision_tower.config.image_size
        self.vision_tower.requires_grad_(self.unfreeze_mm_vision_tower)
        self.is_loaded = True

    @property
    def image_size(self):
        return self._image_size

    def feature_select(self, outputs):
        sequence_output = outputs["last_hidden_state"]  # [B, seq_len, hidden_size]

        if self.select_feature == 'cls_patch':
            image_features = sequence_output
        elif self.select_feature == 'patch':
            image_features = sequence_output[:, 1:]
        elif self.select_feature == 'cls':
            image_features = sequence_output[:, 0]
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    #这个地方没有考虑到如果是多个影像应该返回什么？？？可能是一个bug，但是目前运行起来没问题
    def _forward(self, images):
        with torch.set_grad_enabled(self.unfreeze_mm_vision_tower):
            image_forward_outs = self.vision_tower.forward(images.to(device=self.device, dtype=self.dtype))

            # 缓存token数，只取第一个batch的长度，减1是去除cls token
            if self._num_patches_cached is None:
                seq_len = image_forward_outs['last_hidden_state'].shape[1]
                self._num_patches_cached = seq_len - 1

            image_features = self.feature_select(image_forward_outs).to(images.dtype)
            return image_features, image_forward_outs

    @property
    def num_patches(self):
        if self._num_patches_cached is None:
            raise RuntimeError("Model not forwarded yet, num_patches unknown.")
        return self._num_patches_cached

    @property
    def num_patches_per_side(self):
        return int(self.num_patches ** 0.5)

    @property
    def patch_size(self):
        return self._patch_size

    @property
    def hidden_size(self):
        return self._hidden_size
