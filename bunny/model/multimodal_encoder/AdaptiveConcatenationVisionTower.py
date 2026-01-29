import torch
import torch.nn as nn
import torch.nn.functional as F
from .dino_encoder import DinoVisionTower
from .siglip.siglip_encoder import SiglipVisionTower
from bunny.util.utils import CrossAttentionBlock
from bunny.util.merge import bipartite_soft_matching_merge, random_bipartite_soft_matching
from PIL import Image
from torchvision import transforms
from typing import Dict, List, Union, Optional

class WeightedPseudoCLSHead(nn.Module):
    def __init__(self, dim, hidden_dim_ratio=2):
        super().__init__()
        hidden_dim = int(dim * hidden_dim_ratio)
        self.score_predictor = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, tokens):
        scores = self.score_predictor(tokens.to(dtype=tokens.dtype))
        weights = F.softmax(scores, dim=1) 
        return torch.sum(tokens * weights, dim=1, keepdim=True)

class ImageProcessorMultipleEncoders:
    def __init__(self, patch_size_list: List[int], target_size: int = 384):
        self.target_size = target_size 
        self.patch_lcm = 14 
        self.crop_size = {"height": target_size, "width": target_size}
        self.dino_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.siglip_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def preprocess(self, images, **kwargs):
        new_size = (self.target_size // self.patch_lcm) * self.patch_lcm
        if isinstance(images, torch.Tensor): return {"pixel_values": torch.stack([images, images], dim=1)}
        if not isinstance(images, list): images = [images]
        stacked = []
        for img in images:
            if not isinstance(img, Image.Image): img = Image.open(img).convert('RGB')
            img_res = img.resize((new_size, new_size), Image.BILINEAR)
            stacked.append(torch.stack([self.dino_transform(img_res), self.siglip_transform(img_res)], dim=0))
        return {"pixel_values": torch.stack(stacked).contiguous()}

class AdaptiveConcatenationVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()
        self.is_loaded = False
        self.args = args
        self.global_dimension = getattr(args, "mm_hidden_size", 1024)
        self.compression_K = getattr(args, "compression_K", 8)
        self.num_heads = 8 
        self.mlp_ratio = 4.0
        self.target_image_size = 384
        self.image_processor = ImageProcessorMultipleEncoders([14], self.target_image_size)
             
        self.dino_vision_tower = DinoVisionTower(args.vision_tower_dino, args)
        self.siglip_vision_tower = SiglipVisionTower(args.vision_tower_siglip, args)
        self.mlp_layers = nn.ModuleList([
            nn.Linear(self.dino_vision_tower.hidden_size, self.global_dimension),
            nn.Linear(self.siglip_vision_tower.hidden_size, self.global_dimension)
        ])
        self._init_interaction_modules()
        if not delay_load: self.load_model()

    def _init_interaction_modules(self):
        self.N_layer_A = self.dino_vision_tower.layer_count 
        self.multi_cls_cross_attn_blocks_A = nn.ModuleList([CrossAttentionBlock(self.global_dimension, self.num_heads, self.mlp_ratio) for _ in range(self.N_layer_A)])
        self.dino_cls_attn_weights = nn.Parameter(torch.ones(self.N_layer_A))
        self.N_layer_B = self.siglip_vision_tower.layer_count 
        self.b_pseudo_cls_head = WeightedPseudoCLSHead(self.global_dimension) 
        self.multi_cls_cross_attn_blocks_B = nn.ModuleList([CrossAttentionBlock(self.global_dimension, self.num_heads, self.mlp_ratio) for _ in range(self.N_layer_B)])
        self.oryx_cls_attn_weights = nn.Parameter(torch.ones(self.N_layer_B))
        self.final_cls_weights = nn.Parameter(torch.ones(2))

    def load_model(self):
        if self.is_loaded: return
        self.dino_vision_tower.load_model()
        self.siglip_vision_tower.load_model()
        self.is_loaded = True

    @property
    def dtype(self): return self.mlp_layers[0].weight.dtype
    @property
    def device(self): return self.mlp_layers[0].weight.device

    @property
    def hidden_size(self): return self.global_dimension
    

    def forward(self, images):
        # 1. 影像预处理与维度展开
        if isinstance(images, dict): images = images.get("pixel_values", images)
        target_res = 378
        
        # 统一形状为 (Total_Images, 2, C, H, W)
        if isinstance(images, torch.Tensor):
            if images.dim() == 5:
                B, N, C, H, W = images.shape
                images = images.view(B * N, C, H, W) # 这里 N 可能不等于 2，而是框架处理后的结果
                # 每张图内部再次 stack 出两个编码器需要的输入 (双塔逻辑)
                images = torch.stack([images, images], dim=1) 
            elif images.dim() == 4:
                # 如果只有 (Total_Images, C, H, W)
                images = torch.stack([images, images], dim=1)
        else:
            images = self.image_processor.preprocess(images)["pixel_values"]

        # 强制拉伸到 378 对齐位置编码
        if images.shape[-1] != target_res:
            B_total, N_enc, C, H, W = images.shape
            images = images.view(B_total * N_enc, C, H, W)
            images = F.interpolate(images.to(torch.float32), size=(target_res, target_res), mode='bilinear').to(self.dtype)
            images = images.view(B_total, N_enc, C, target_res, target_res)

        images = images.to(device=self.device, dtype=self.dtype)
        # 此时 Total_B 是框架预期的 split_sizes 总和
        Total_B = images.shape[0] 
        
        dino_input, siglip_input = images[:, 0].contiguous(), images[:, 1].contiguous()

        # 2. 视觉特征提取 (对所有图并行处理)
        _, A_inter = self.dino_vision_tower(dino_input) 
        _, B_inter = self.siglip_vision_tower(siglip_input) 
        
        A_proj = self.mlp_layers[0](A_inter.to(self.dtype))
        B_proj = self.mlp_layers[1](B_inter.to(self.dtype)) 

        # 3. 动态维度拆分
        L_per_layer = A_proj.shape[1] // self.N_layer_A
        T_actual = L_per_layer - 1 

        A_full = A_proj.view(Total_B, self.N_layer_A, L_per_layer, -1)
        B_full = B_proj.view(Total_B, self.N_layer_B, L_per_layer, -1)
        A_cls, A_patches = A_full[:, :, 0:1, :], A_full[:, :, 1:, :] 
        B_patches = B_full[:, :, 1:, :] 

        # 4. 压缩逻辑 (K=8)
        target_context = T_actual // self.compression_K
        
        # 处理压缩
        def get_kv_ctx(p):
            p_flat = p.flatten(0, 1).contiguous()
            m = bipartite_soft_matching_merge(p_flat, target_context, p_flat)
            return m.view(Total_B, -1, self.global_dimension)

        B_kv_context = get_kv_ctx(B_patches)
        A_kv_context = get_kv_ctx(A_patches)

        # 交互增强
        def cross_attn_group(cls_t, kv_t, blocks, weights):
            enhanced = []
            for i in range(len(blocks)):
                enhanced.append(blocks[i](torch.cat([cls_t[:, i], kv_t], dim=1))[:, 0:1, :])
            return torch.sum(torch.cat(enhanced, dim=1) * F.softmax(weights, dim=0).view(1, -1, 1).to(self.dtype), dim=1, keepdim=True)

        final_cls_A = cross_attn_group(A_cls, B_kv_context, self.multi_cls_cross_attn_blocks_A, self.dino_cls_attn_weights)
        
        # B塔伪CLS逻辑
        pseudo_cls_B_all = self.b_pseudo_cls_head(B_kv_context.unsqueeze(1).repeat(1, self.N_layer_B, 1, 1).flatten(0, 1)).view(Total_B, self.N_layer_B, 1, -1)
        final_cls_B = cross_attn_group(pseudo_cls_B_all, A_kv_context, self.multi_cls_cross_attn_blocks_B, self.oryx_cls_attn_weights)

        enhanced_cls_token = torch.sum(torch.cat([final_cls_A, final_cls_B], dim=1) * F.softmax(self.final_cls_weights, dim=0).view(1, 2, 1).to(self.dtype), dim=1, keepdim=True)

        # 5. 拼接 365 Tokens
        A_up, A_lo = A_patches[:, :self.N_layer_A//2].flatten(1, 2), A_patches[:, self.N_layer_A//2:].flatten(1, 2)
        B_up, B_lo = B_patches[:, :self.N_layer_B//2].flatten(1, 2), B_patches[:, self.N_layer_B//2:].flatten(1, 2)

        def merge_to(x, n):
            if x.shape[1] <= n: return x 
            m_f, _ = random_bipartite_soft_matching(x, r=x.shape[1]-n)
            return m_f(x)

        target_n = T_actual // self.compression_K 
        # 结果形状必须是 (Total_B, 365, 1024)
        out = torch.cat([enhanced_cls_token, merge_to(A_up, target_n), merge_to(B_up, target_n), merge_to(A_lo, target_n), merge_to(B_lo, target_n)], dim=1)
        
        return out, None