import torch
import torch.nn.functional as F
from modelscope import AutoImageProcessor, AutoModel, AutoConfig

from .base_encoder import BaseVisionTower


def extract_res_interp(model_name):
    valid_model_prefixes = [
        "facebook/dinov3-convnext-large-pretrain-lvd1689m", #大型
        "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",   #小型
    ]

    for prefix in valid_model_prefixes:
        if model_name.startswith(prefix):
            base_model_name = prefix
            break
    else:
        raise ValueError(f"Unknown vision tower: {model_name}")

    res = 224   # 这两个分辨率都是一样大
    interp = None

    return base_model_name, res, interp


class DinoVisionTower(BaseVisionTower):
    def __init__(self, vision_tower, args, delay_load=False):
        super(DinoVisionTower, self).__init__(vision_tower, args, delay_load)
        base_model_name, res, interp = extract_res_interp(self.vision_tower_name)
        self._vision_tower_name = vision_tower
        self.vision_tower_name = base_model_name
        self._image_size = res
        self._interp_size = interp
        self._patch_size = 16  # default patch size

        if not self.delay_load:
            self.load_model()
        else:
            self.cfg_only = AutoConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):

   # 加载预训练模型和处理器
        self.vision_tower = AutoModel.from_pretrained(self.vision_tower_name)
        self.image_processor = AutoImageProcessor.from_pretrained(self.vision_tower_name)
        # 读取模型配置参数
        #self._patch_size = self.vision_tower.stages[0].downsample_layers[0].stride[0]  # 4
        self._hidden_size = self.vision_tower.layer_norm.normalized_shape[0]  # 1536
        self._image_size = self.vision_tower.config.image_size  # 224
        self.vision_tower.requires_grad_(self.unfreeze_mm_vision_tower)
        self.is_loaded = True

    @property
    def image_size(self):
        return self._image_size

    def feature_select(self, outputs):
        sequence_output = outputs["last_hidden_state"]  # batch_size, sequence_length, hidden_size

        if self.select_feature == 'cls_patch':
            image_features = sequence_output
        elif self.select_feature == 'patch':
            image_features = sequence_output[:, 1:]
        elif self.select_feature == 'cls':
            image_features = sequence_output[:, 0]
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    def interpolate(self, image_features):
        if self._interp_size is None:
            return image_features

        b, num_tokens, dim = image_features.shape

        if num_tokens != self.num_patches:
            target_h = target_w = int(self._interp_size ** 0.5)
            h = w = int(num_tokens ** 0.5)

            image_features = image_features.view(b, h, w, dim)
            image_features = image_features.permute(0, 3, 1, 2).contiguous()

            image_features = F.interpolate(
                image_features.to(torch.float32),
                size=(target_h, target_w),
                mode='bilinear',
                align_corners=False
            ).to(image_features.dtype)

            # Permute the dimensions back to (b, target_h, target_w, dim)
            image_features = image_features.permute(0, 2, 3, 1).contiguous()

            # Flatten the spatial dimensions (target_h, target_w) into a single dimension
            image_features = image_features.flatten(1, 2)

        return image_features

    def _forward(self, images):
        # logger.warning(f"images shape: {images.shape}")
        with torch.set_grad_enabled(self.unfreeze_mm_vision_tower):
            image_forward_outs = self.vision_tower.forward(images.to(device=self.device, dtype=self.dtype))
            image_features = self.feature_select(image_forward_outs).to(images.dtype)
            return image_features

    @property
    def num_patches_per_side(self):
        return int(self.num_patches ** 0.5)

    @property
    def num_patches(self):
        if self._interp_size is None:
            return (self._image_size // self._patch_size) ** 2
        else:
            return self._interp_size


    @property
    def hidden_size(self):
        return self._hidden_size