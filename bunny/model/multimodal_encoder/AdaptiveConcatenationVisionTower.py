import torch
import torch.nn as nn
import torch.nn.functional as F
from .dino_encoder import DinoVisionTower
from .siglip.siglip_encoder import SiglipVisionTower
from bunny.util.utils import CrossAttentionBlock
from bunny.util.merge import bipartite_soft_matching_merge, random_bipartite_soft_matching
from PIL import Image
from typing import Dict, List, Union, Optional

class FoveatedAnchorSampler(nn.Module):
    def __init__(self, embed_dim=1024):
        super().__init__()
        self.grid_size = 24    # 单图 24x24 patches
        self.full_grid = 48    # 4角落拼接 48x48
        # S2C: Space-to-Channel 无损压缩
        self.s2c_projector = nn.Sequential(
            nn.Linear(embed_dim * 4, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )        
        # 🔥 结构感知：Depthwise Conv 捕获表格线条/笔画方向
        self.struct_extractor = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim),
            nn.GroupNorm(32, embed_dim),
            nn.GELU()
        )
        # 显著性评分器
        self.scorer = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, 1)
        )
        # 🔥 OCR中心偏置权重（可选，表格文本密集区）
        self.center_weight = nn.Parameter(torch.ones(1, 144, 1))

    def get_rope_style_pos(self, x):
        B, N, C = x.shape
        device, dtype = x.device, x.dtype
        grid_size = int(N ** 0.5 + 0.5) 
        
        coords = torch.linspace(-1, 1, grid_size, device=device, dtype=dtype)
        grid_h, grid_w = torch.meshgrid(coords, coords, indexing='ij')
        
        # 结果为 (H, W, 2)，展平后为 [N, 2]
        pos_guide = torch.stack([grid_h, grid_w], dim=-1).reshape(1, -1, 2).expand(B, -1, -1)
        return pos_guide
    
    def forward(self, center_feat, full_feat):
        B, N_center, C = center_feat.shape  # [B, 576, 1024]
        N_full = full_feat.shape[1]         # 2304
        # === 中心区域：结构增强 + S2C压缩 ===
        x_2d = center_feat.view(B, 24, 24, C).permute(0, 3, 1, 2)  # [B,C,24,24]
        x_struct = self.struct_extractor(x_2d)                     # 结构卷积
        x_2d = x_struct.permute(0, 2, 3, 1).reshape(B, 576, C)     # [B,576,C]
        # S2C下采样：24x24 → 12x12 (144 tokens)
        x_s2c = x_2d.view(B, 12, 2, 12, 2, C).permute(0, 1, 3, 2, 4, 5).contiguous()
        x_s2c = x_s2c.view(B, 144, C * 4)
        center_base = self.s2c_projector(x_s2c) * self.center_weight  # 中心偏置
        # === 显著性采样：位置感知 ===
        pos_guide = self.get_rope_style_pos(full_feat)  # [B,2304,2]
        # 评分 + 距离惩罚（中心优先）
        raw_scores = self.scorer(full_feat).squeeze(-1)  # [B,2304]
        dist_penalty = torch.norm(pos_guide, dim=-1, p=2) * 0.1
        scores = raw_scores - dist_penalty
        # Top-210采样（表格线条/文字密集区）
        _, top_indices = torch.topk(scores, k=210, dim=1, sorted=True)
        batch_indices = torch.arange(B, device=full_feat.device).unsqueeze(1).expand(-1, 210)
        selected_patches = full_feat[batch_indices, top_indices]
        return center_base, selected_patches  # [B,144,C], [B,210,C]
    

class ImageProcessorMultipleEncoders:
    def __init__(self, patch_size_list: List[int] = [14], target_size: int = 378):
        # 强制锁定 378，因为这是 14 patch_size 的最佳倍数 (14 * 27 = 378)
        self.target_size = 378 
        self.patch_lcm = 14 
        self.dino_transform = None # 在 runtime 初始化
        self.siglip_transform = None
    def preprocess(self, images, return_tensors='pt', **kwargs):
        """
        输入: List[PIL.Image] 或 单个 PIL.Image
        输出: {'pixel_values': tensor [N, 2, 3, 378, 378]}
        """
        if not isinstance(images, list): images = [images]
        # 懒加载 torchvision transform 以避免多进程 Pickle 问题
        if self.dino_transform is None:
            from torchvision import transforms
            mean_dino = (0.485, 0.456, 0.406)
            std_dino = (0.229, 0.224, 0.225)
            mean_siglip = (0.5, 0.5, 0.5)
            std_siglip = (0.5, 0.5, 0.5)
            self.dino_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean_dino, std_dino),
            ])
            self.siglip_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean_siglip, std_siglip),
            ])
        stacked = []
        for img in images:
            if not isinstance(img, Image.Image): 
                # 防止传入路径字符串
                img = Image.open(img).convert('RGB')
            # 1. 强制 Resize 到 378x378
            img_res = img.resize((self.target_size, self.target_size), Image.BILINEAR)
            # 2. 双路处理并 Stack
            # 结果形状 [2, 3, 378, 378]
            dual_tower_tensor = torch.stack([
                self.dino_transform(img_res), 
                self.siglip_transform(img_res)
            ], dim=0)
            stacked.append(dual_tower_tensor)
        # 最终形状 [N_images, 2, 3, 378, 378]
        return {"pixel_values": torch.stack(stacked).contiguous()}
    
class AdaptiveConcatenationVisionTower(nn.Module):
    def __init__(self, vision_tower, args,  **kwargs):  #参数要透传到子模块中去
        super().__init__()
        self.is_loaded = False
        self.training_stage = kwargs.get('training_stage', getattr(args, 'training_stage', 'inference'))  
        print(f"🎨 [MixedEncoder] 成功识别training_stage: {self.training_stage}")
        self.delay_load = kwargs.get('delay_load', False) 
        print(f"🎨 [MixedEncoder] 成功识别delay_load: {self.delay_load}")    
        self.args = args
        self.global_dimension = getattr(args, "mm_hidden_size", 1024)
        self.compression_K = getattr(args, "compression_K", 8)
        self.num_heads = 8 
        self.mlp_ratio = 4.0
        self.target_image_size = 378
        self.image_processor = ImageProcessorMultipleEncoders()
        self.unfreeze_mm_vision_tower = getattr(args, 'unfreeze_mm_vision_tower', False)
        self.args = args
        self.dino_vision_tower = DinoVisionTower(args.vision_tower_dino, args, **kwargs)
        self.siglip_vision_tower = SiglipVisionTower(args.vision_tower_siglip, args, **kwargs)
        self.mlp_layers = nn.ModuleList([
            nn.Linear(self.dino_vision_tower.hidden_size, self.global_dimension),
            nn.Linear(self.siglip_vision_tower.hidden_size, self.global_dimension)
        ])
        self.dino_projector = nn.Linear(768, 1024)   #两个视觉编码器融合前需要对齐维度数，dino通常返回的是768维度
        self.siglip_projector = nn.Linear(1152, 1024)  ##两个视觉编码器融合前需要对齐维度数，siglip通常返回的是1152维度
        self.view_tag_embedding = nn.Embedding(6, self.global_dimension)
        self.saliency_sampler = FoveatedAnchorSampler(embed_dim=self.global_dimension)
        self.cross_attn_dino_q = nn.ModuleList([
            CrossAttentionBlock(dim=1024, num_heads=self.num_heads) for _ in range(self.dino_vision_tower.layer_count)
        ])
        # path_B: SigLIP as Query
        self.cross_attn_siglip_q = nn.ModuleList([
            CrossAttentionBlock(dim=1024, num_heads=self.num_heads) for _ in range(self.dino_vision_tower.layer_count)
        ])   
        #针对于全局图的多层自适应增强cls token
        self.gate_mlps = nn.ModuleList([
            nn.Sequential(
            nn.Linear(2048, 512),
            nn.GELU(),
            nn.Linear(512, 1) # 输出标量权重
            ) for _ in range(self.dino_vision_tower.layer_count)
        ])
        self.feat_norm = nn.LayerNorm(self.global_dimension)  
        self.super_anchor_proj = nn.Linear(self.global_dimension * 2, self.global_dimension)
        nn.init.zeros_(self.super_anchor_proj.weight)
        nn.init.zeros_(self.super_anchor_proj.bias)
        self.pixel_fusion_gate = nn.Sequential(
                                    nn.Linear(self.global_dimension * 2, self.global_dimension // 4),
                                    nn.GELU(),
                                    nn.Linear(self.global_dimension // 4, 1),
                                    nn.Sigmoid() 
                                )
        if not self.delay_load: 
            self.load_model()
    def _set_subtower_grad_state(self):
        """统一管理子塔的梯度和模式状态"""
        # 这里的打印直接说明当前的业务意图
        mode_desc = "🚀 [全量微调/全参数模式]" if self.unfreeze_mm_vision_tower else "🔒 [冻结模式/只读推理模式]"
        print(f"🛠️  [MixedEncoder 属性设定] 业务意图: {mode_desc}")
        is_actually_unfreezing = (self.training_stage == 'finetune') and self.unfreeze_mm_vision_tower
        for sub_tower in [self.siglip_vision_tower, self.dino_vision_tower]:
            if sub_tower is not None:
                # 注入属性
                if hasattr(sub_tower, 'config'):
                    sub_tower.config.unfreeze_mm_vision_tower = is_actually_unfreezing
                sub_tower.unfreeze_mm_vision_tower = is_actually_unfreezing
                # 获取名字，如果是 DINO 或 SigLIP 应该能看出来
                t_name = getattr(sub_tower, "vision_tower_name", "Sub-Tower")
                if is_actually_unfreezing:
                    sub_tower.requires_grad_(True)
                    sub_tower.train()
                    print(f"   💡 子塔 {t_name}: 已解锁权重。它将随主模型一起更新（微调必备）。")
                else:
                    sub_tower.requires_grad_(False)
                    sub_tower.eval()
                    print(f"   💡 子塔 {t_name}: 已锁定权重。它将作为纯特征提取器使用（预训练/推理必备）。")
    def load_model(self):
        if self.is_loaded:
            return
        self.dino_vision_tower.load_model()
        self.siglip_vision_tower.load_model()
        self._set_subtower_grad_state()
        self.is_loaded = True
    @property
    def dtype(self): return self.mlp_layers[0].weight.dtype
    @property
    def device(self): return self.mlp_layers[0].weight.device
    @property
    def hidden_size(self): return self.global_dimension
    def forward(self, images):
        try:
            rank = torch.distributed.get_rank()
        except Exception:
            rank = 0 # 如果不是分布式训练，默认为 0
        # 只有 Rank 0 负责“发声”
        if rank == 0:
            if not hasattr(self, "has_printed_shape"):
                print("\n" + "👁️" * 15 + " RANK 0 独家质检 " + "👁️" * 15)
                print(f"🚀 [VISION TOWER ENTRY]")
                print(f"   - Images Shape: {images.shape}")
                # 顺便检查一下数据在哪张卡上
                print(f"   - Device: {images.device}") 
                # 检查一下数值范围，确保没有溢出或全零
                print(f"   - Mean Value: {images.mean().item():.4f}") 
                print("👁️" * 40 + "\n")
            self.has_printed_shape = True
        device = images.device
        self.siglip_vision_tower.to(device)
        self.dino_vision_tower.to(device)
        b, num_crops, num_towers, c, h, w = images.shape
        dino_input = images[:, :, 0]   # [B, 6, 3, 378, 378]
        siglip_input = images[:, :, 1] # [B, 6, 3, 378, 378]
        dino_input = dino_input.view(-1, c, h, w)
        siglip_input = siglip_input.view(-1, c, h, w)
        dino_out, dino_gallery = self.dino_vision_tower(dino_input)
        siglip_out, siglip_gallery = self.siglip_vision_tower(siglip_input)
        dino_gallery = dino_gallery.view(b, num_crops, -1, dino_gallery.shape[-1]) #再转换成[B,6,4*577,768]
        siglip_gallery = siglip_gallery.view(b, num_crops, -1, siglip_gallery.shape[-1]) #优雅的转换回来[B,6,8*577,1152]
        all_dino_feats = self.dino_projector(dino_gallery) 
        all_siglip_feats = self.siglip_projector(siglip_gallery)
        g_dino_feat = all_dino_feats[:, 0]  ## [B, 2308, 1024]
        g_siglip_feat = all_siglip_feats[:, 0]  # [B, 4616, 1024]
        B, _, D_common = g_dino_feat.shape   #都转换成1024维度
        dino_layers = g_dino_feat.view(B, self.dino_vision_tower.layer_count, 577, D_common)
        dino_cls_tokens = dino_layers[:, :, 0:1, :]   # [B, 4, 1, D] -> 这是 DINO 的“指挥官”
        dino_patches    = dino_layers[:, :, 1:, :]    # [B, 4, 576, D] -> 这是 DINO 的“躯干”
        siglip_layers = g_siglip_feat.view(B, self.siglip_vision_tower.layer_count, 577, D_common)
        siglip_cls_tokens = siglip_layers[:, :, 0:1, :] # [B, 8, 1, D] -> 这是 SigLIP 的“伪指挥官”
        siglip_patches    = siglip_layers[:, :, 1:, :]  # [B, 8, 576, D] -> 这是 SigLIP 的“躯干”
        #使用层对层对层（Layer-to-Layer） 是为了保证 Query 和 Key 在“语义高度”上是相对匹配的。
        enhanced_global_tokens_list = []
        for i in range(4):
            # === 准备数据 ===
            # DINO 方：取第 i 层
            curr_dino_cls = dino_cls_tokens[:, i, :, :]  # Query A: [B, 1, D]
            curr_dino_pat = dino_patches[:, i, :, :]     # Key/Value B: [B, 576, D]
            # SigLIP 方：取第 2*i 和 2*i+1 层 (2对1策略)
            # 将两层的 Patch 拼起来，提供更丰富的信息源
            curr_siglip_pat = torch.cat([
                siglip_patches[:, 2*i, :, :], 
                siglip_patches[:, 2*i+1, :, :]
            ], dim=1) # Key/Value A: [B, 576*2, D]
            # 将两层的 CLS 平均一下，做一个超级语义 Query
            curr_siglip_cls = (siglip_cls_tokens[:, 2*i, :, :] + siglip_cls_tokens[:, 2*i+1, :, :]) * 0.5
            # Query B: [B, 1, D]
            combined_dino_in = torch.cat([curr_dino_cls, curr_siglip_pat], dim=1)
            # === 交互 A: DINO 主动吸收 SigLIP 语义 ===
            # DINO CLS 问 SigLIP Patches: "这里面是什么物体？"
            # self.cross_attn_dino_q[i] 是一个 CrossAttentionBlock
            dino_enhanced = self.cross_attn_dino_q[i](combined_dino_in) # [B, 1, D]
            combined_siglip_in = torch.cat([curr_siglip_cls, curr_dino_pat], dim=1)
            # === 交互 B: SigLIP 借用 DINO 骨架 ===
            # SigLIP CLS 问 DINO Patches: "这个物体边界在哪？"
            siglip_enhanced = self.cross_attn_siglip_q[i](combined_siglip_in) # [B, 1, D]
            # === 动态权重融合 (Adaptive Gating) ===
            # 拼接两者，计算一个 0~1 的权重 alpha
            gate_input = torch.cat([dino_enhanced, siglip_enhanced], dim=-1) # [B, 1, 2*D]
            alpha = torch.sigmoid(self.gate_mlps[i](gate_input)) # [B, 1, 1]
            # 融合：得到这一层最强的 Global Token
            combined_token = alpha * dino_enhanced + (1 - alpha) * siglip_enhanced
            enhanced_global_tokens_list.append(combined_token)
        final_global_cls_tokens = torch.cat(enhanced_global_tokens_list, dim=1)  #[B, 4, 1024]
        if rank == 0 and not hasattr(self, "has_printed_fusion"):
            #print(f"🔥 [FUSION] Global CLS Fusion Complete. Shape: {final_global_cls_tokens.shape}")
            self.has_printed_fusion = True
        # 1. 获取通用维度信息
        B = all_dino_feats.shape[0]
        D_common = 1024  # 投影后的统一维度
        num_patches = 576 # 24x24 (不含 CLS)
        total_tokens_per_layer = num_patches + 1 # 577 (含 CLS)
        # 获取动态层数 (这是关键！DINO是4，SigLIP是8)
        num_layers_dino = self.dino_vision_tower.layer_count   # 4
        num_layers_siglip = self.siglip_vision_tower.layer_count # 8
        # ---------------------------------------------------------
        # A. 处理中间子图 (Center Crop, Index 5)
        # ---------------------------------------------------------
        # [A1] 提取 DINO 中间图
        # 原始数据: [B, 4*577, 1024] -> Reshape 为 [B, 4层, 577个, 1024]
        center_dino_raw = all_dino_feats[:, 5]
        center_dino_reshaped = center_dino_raw.view(B, num_layers_dino, total_tokens_per_layer, D_common)
        # [A2] 提取 SigLIP 中间图 (注意这里用 num_layers_siglip = 8)
        # 原始数据: [B, 8*577, 1024] -> Reshape 为 [B, 8层, 577个, 1024]
        center_siglip_raw = all_siglip_feats[:, 5]
        center_siglip_reshaped = center_siglip_raw.view(B, num_layers_siglip, total_tokens_per_layer, D_common)
        # [A3] 精准切片：只取各自的“最后一层” + “纯 Patch”
        # DINO: 取第 4 层 (index -1), 去掉第一个 CLS (index 1:)
        c_dino_last = center_dino_reshaped[:, -1, 1:, :]   # [B, 576, 1024]
        # SigLIP: 取第 8 层 (index -1), 去掉第一个 CLS (index 1:)
        c_siglip_last = center_siglip_reshaped[:, -1, 1:, :] # [B, 576, 1024]
        # [A4] 融合 (现在两者都是 [B, 576, 1024]，物理空间完全对齐)
        #center_feat_fused = (c_dino_last + c_siglip_last) * 0.5
        fusion_input = torch.cat([c_dino_last, c_siglip_last], dim=-1)
        alpha_center = self.pixel_fusion_gate(fusion_input)
        center_feat_fused = alpha_center * c_dino_last + (1 - alpha_center) * c_siglip_last
        # ---------------------------------------------------------
        # B. 处理四个角落子图 (Corner Crops, Index 1-4)
        # ---------------------------------------------------------
        # [B1] 提取 DINO 角落图
        # 原始数据: [B, 4张图, 4*577, 1024]
        corners_dino_raw = all_dino_feats[:, 1:5]
        # Reshape: [B, 4张图, 4层, 577个, 1024]
        corners_dino_reshaped = corners_dino_raw.view(B, 4, num_layers_dino, total_tokens_per_layer, D_common)
        corners_siglip_raw = all_siglip_feats[:, 1:5]
        corners_siglip_reshaped = corners_siglip_raw.view(B, 4, num_layers_siglip, total_tokens_per_layer, D_common)
        corners_dino_last = corners_dino_reshaped[:, :, -1, 1:, :]   # [B, 4, 576, 1024]
        corners_siglip_last = corners_siglip_reshaped[:, :, -1, 1:, :] # [B, 4, 576, 1024]
       
        corner_input  = torch.cat([corners_dino_last, corners_siglip_last], dim=-1)
        alpha_corner = self.pixel_fusion_gate(corner_input)
        corner_feats_fused = alpha_corner * corners_dino_last + (1 - alpha_corner) * corners_siglip_last

        anchor_center = center_feat_fused.mean(dim=1, keepdim=True)
        anchor_corners = corner_feats_fused.mean(dim=2)
        g_dino_last = dino_layers[:, -1, 1:, :]     # [B, 576, 1024]
        g_siglip_last = siglip_layers[:, -1, 1:, :] # [B, 576, 1024]
        g_fusion_input = torch.cat([g_dino_last, g_siglip_last], dim=-1)
        alpha_g = self.pixel_fusion_gate(g_fusion_input)
        g_feat_fused = alpha_g * g_dino_last + (1 - alpha_g) * g_siglip_last
        anchor_global = g_feat_fused.mean(dim=1, keepdim=True)
        view_ids = torch.arange(6, device=device)
        view_tags = self.view_tag_embedding(view_ids) # [6, 1024]
        view_tags = view_tags.unsqueeze(0).expand(B, -1, -1) # 扩展到 Batch 维度 [B, 6, 1024]
        # 将 6 个视角的摘要拼接在一起 -> [B, 6, 1024]
        # 顺序：[0:全局, 1-4:角落, 5:中心]
        # 将 6 个视角的摘要拼接
        raw_anchors = torch.cat([anchor_global, anchor_corners, anchor_center], dim=1)
        # 核心修改：让 Anchor 知道自己代表哪个位置
        view_anchors = raw_anchors + view_tags

        B, _, _, D = corner_feats_fused.shape # corner_feats_fused: [B, 4, 576, 1024]
        corners = corner_feats_fused.view(B, 2, 2, 24, 24, D)
        full_grid_feat = corners.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, 48, 48, D)
        full_feat_flattened = full_grid_feat.view(B, 2304, D)

        # ---------------------------------------------------------
        # C. 召唤采样器 (现在输入非常纯净且正确)
        # ---------------------------------------------------------
        center_base, selected_patches = self.saliency_sampler(
            center_feat_fused, 
            full_feat_flattened
        )
        B = final_global_cls_tokens.shape[0]
        device = final_global_cls_tokens.device
        dtype = final_global_cls_tokens.dtype
        # =================================================================
        # 🌟 [新增模块 B]：生成 Super-Fusion Anchor (第 365 个 Token)
        # =================================================================
        # 取那 4 个跨塔对齐的全局 CLS 的均值
        main_cls_avg = final_global_cls_tokens.mean(dim=1, keepdim=True) # [B, 1, 1024]
        # 取这 6 个视角的均值
        all_view_avg = view_anchors.mean(dim=1, keepdim=True)            # [B, 1, 1024]
        combined_meta = torch.cat([main_cls_avg, all_view_avg], dim=-1)  # [B, 1, 2048]
        super_anchor = self.super_anchor_proj(combined_meta)             # [B, 1, 1024]
        final_embeddings = torch.cat([
            final_global_cls_tokens,  # Token 0-3   (4个：双塔深度交互后的核心语义)
            view_anchors,             # Token 4-9   (6个：全局+4角落+中心的宏观路标)
            center_base,              # Token 10-153(144个：中心区域的基础骨架)
            selected_patches,         # Token 154-363(210个：角落区域的高精采样)
            super_anchor              # Token 364   (1个：全文视觉大一统总结)
        ], dim=1)
        # 必杀技：通过 LayerNorm 抹平各路特征的尺度差异，防止 Loss 回弹
        final_embeddings = self.feat_norm(final_embeddings)
        #print(f"🚀 [混合塔返回特征] Final Embedding Shape:    {final_embeddings.shape}")
        return final_embeddings, None