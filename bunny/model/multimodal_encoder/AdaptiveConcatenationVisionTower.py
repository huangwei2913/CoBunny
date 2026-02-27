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



class FoveatedAnchorSampler(nn.Module):
    def __init__(self, embed_dim=1024):
        super().__init__()
        # --- 核心改动：适配强制对齐后的 24x24 特征网格 ---
        self.grid_size = 24   # 对应 378 分辨率下被 interpolate 后的特征大小
        self.full_grid = 48   # 24 * 2 = 48 (2x2 拼接的大图)
        
        # 定义 48x48 的全局位置编码表
        self.global_pos_embed = nn.Parameter(torch.randn(1, self.full_grid**2, embed_dim) * 0.02)
        
        # 评分器：决定哪些 Patch 值得留下
        # 改进：输入 1024 -> 映射 -> 1 (Score)
        self.scorer = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, 1)
        )

    def get_crop_pos_embeds(self):
        """
        在 48x48 的棋盘上切出 5 个子图的位置编码
        """
        B = 1
        grid = self.global_pos_embed.view(B, self.full_grid, self.full_grid, -1) # [1, 48, 48, D]
        G = self.grid_size # 24
        
        # 1. 四个角落 (严丝合缝覆盖 48x48)
        pos_tl = grid[:, 0:G, 0:G, :].reshape(B, G*G, -1)   # 左上
        pos_tr = grid[:, 0:G, G:, :].reshape(B, G*G, -1)    # 右上
        pos_bl = grid[:, G:, 0:G, :].reshape(B, G*G, -1)    # 左下
        pos_br = grid[:, G:, G:, :].reshape(B, G*G, -1)     # 右下
        
        # 2. 中间子图 (Center Crop)
        # 逻辑：大图中心点是 24。
        # 24x24 的中心图范围是 [24-12 : 24+12] = [12 : 36]
        pos_center = grid[:, 12:36, 12:36, :].reshape(B, G*G, -1)
        
        return pos_tl, pos_tr, pos_bl, pos_br, pos_center

    def forward(self, center_feat, corner_feats, num_topk=210):
        """
        center_feat: [B, 576, 1024] (不含 CLS)
        corner_feats: [B, 4, 576, 1024] (不含 CLS)
        """
        B = center_feat.shape[0]
        # 获取坐标纹身
        p_tl, p_tr, p_bl, p_br, p_center = self.get_crop_pos_embeds()
        
        # --- A. 注入坐标 (让 Patch 知道自己的物理位置) ---
        center_feat = center_feat + p_center
        
        # 给 4 个角落分别加坐标 (注意 corner_feats 是 4 维张量)
        # corner_feats: [B, 4, 576, D]
        corner_feats[:, 0] += p_tl
        corner_feats[:, 1] += p_tr
        corner_feats[:, 2] += p_bl
        corner_feats[:, 3] += p_br
        
        # --- B. 锚点准备 (中间图压成 12x12 = 144 Token) ---
        # 24x24 -> pooling -> 12x12 (完美的 2 倍下采样)
        center_base = F.adaptive_avg_pool2d(
            center_feat.transpose(1, 2).view(B, -1, 24, 24), (12, 12)
        ).flatten(2).transpose(1, 2) # [B, 144, 1024]

        # --- C. 筛选角落的“价值地块” ---
        all_corners = corner_feats.view(B, -1, 1024) # [B, 4*576=2304, 1024]
        
        # 评分机制：Center-Guided Saliency
        # 1. 计算中心图的全局摘要
        center_summary = center_feat.mean(dim=1, keepdim=True) # [B, 1, 1024]
        
        # 2. 自身显著性 (Self-Saliency): 这个 Patch 本身包含丰富信息吗？
        saliency_scores = self.scorer(all_corners).squeeze(-1) # [B, 2304]
        
        # 3. 中心相关性 (Relevance): 这个 Patch 和中心图有关联吗？
        # 使用简单的点积注意力
        relevance_scores = torch.matmul(all_corners, center_summary.transpose(1, 2)).squeeze(-1) # [B, 2304]
        
        # 4. 融合分数 (你可以调整权重，这里 1:1)
        final_scores = saliency_scores + relevance_scores

        # Top-K 采样
        _, top_indices = torch.topk(final_scores, k=num_topk, dim=1)
        
        # 提取高价值 Patch (带着坐标纹身一起提取)
        selected_patches = torch.gather(
            all_corners, 1, top_indices.unsqueeze(-1).expand(-1, -1, 1024)
        ) # [B, 210, 1024]
        
        return center_base, selected_patches



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
        self.n_sep1_embed = nn.Parameter(torch.randn(1, 1, 1024) * 0.02)
        self.n_sep2_embed = nn.Parameter(torch.randn(1, 1, 1024) * 0.02)
        self.n_end_embed  = nn.Parameter(torch.randn(1, 5, 1024) * 0.02)
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

        ###############################################################################################
        # 判断传递过来的输入是否符合要求
        # #############################################################################################        
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
        ###############################################################################################
        # 得到4个全局增强cls tokens
        # #############################################################################################        
        b, num_crops, num_towers, c, h, w = images.shape
        dino_input = images[:, :, 0]   # [B, 6, 3, 378, 378]
        siglip_input = images[:, :, 1] # [B, 6, 3, 378, 378]
        
        dino_input = dino_input.view(-1, c, h, w)
        siglip_input = siglip_input.view(-1, c, h, w)

        # 4. 一次性喂给视觉塔 (GPU 会并行的处理这 B*6 张图)
        # dino_out: [B*6, 577, 768] (假设返回的是最后一层)
        # dino_gallery: [B*6, 4*577, 768] (假设是 4 层索引库)
        dino_out, dino_gallery = self.dino_vision_tower(dino_input)

        # siglip_out: [B*6, 577, 1152](这里返回的是最后一层)
        # siglip_gallery: [B*6, 8*577, 1152](假设是 8 层索引库)
        siglip_out, siglip_gallery = self.siglip_vision_tower(siglip_input)

        dino_gallery = dino_gallery.view(b, num_crops, -1, dino_gallery.shape[-1]) #再转换成[B,6,4*577,768]
        siglip_gallery = siglip_gallery.view(b, num_crops, -1, siglip_gallery.shape[-1]) #优雅的转换回来[B,6,8*577,1152]

        all_dino_feats = self.dino_projector(dino_gallery) 
        all_siglip_feats = self.siglip_projector(siglip_gallery)

        g_dino_feat = all_dino_feats[:, 0]  ## [B, 2308, 1024]
        g_siglip_feat = all_siglip_feats[:, 0]  # [B, 4616, 1024]

        # --- 3. 结构重组 (Reshape to Layers) ---
        # DINOv3 (4层, 每层 1个CLS + 576 Patches = 577)
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

        ###############################################################################################
        # 第二个阶段
        # #############################################################################################        
        # =========================================================
        # 🔍 阶段二：肉体采样准备 (Sampling Preparation)
        # =========================================================

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
        center_feat_fused = (c_dino_last + c_siglip_last) * 0.5


        # ---------------------------------------------------------
        # B. 处理四个角落子图 (Corner Crops, Index 1-4)
        # ---------------------------------------------------------

        # [B1] 提取 DINO 角落图
        # 原始数据: [B, 4张图, 4*577, 1024]
        corners_dino_raw = all_dino_feats[:, 1:5]
        # Reshape: [B, 4张图, 4层, 577个, 1024]
        corners_dino_reshaped = corners_dino_raw.view(B, 4, num_layers_dino, total_tokens_per_layer, D_common)

        # [B2] 提取 SigLIP 角落图
        # 原始数据: [B, 4张图, 8*577, 1024]
        corners_siglip_raw = all_siglip_feats[:, 1:5]
        # Reshape: [B, 4张图, 8层, 577个, 1024] (这里必须用 8)
        corners_siglip_reshaped = corners_siglip_raw.view(B, 4, num_layers_siglip, total_tokens_per_layer, D_common)

        # [B3] 精准切片
        # DINO: 取最后一层, 去掉 CLS
        corners_dino_last = corners_dino_reshaped[:, :, -1, 1:, :]   # [B, 4, 576, 1024]

        # SigLIP: 取最后一层, 去掉 CLS
        corners_siglip_last = corners_siglip_reshaped[:, :, -1, 1:, :] # [B, 4, 576, 1024]

        # [B4] 融合
        corner_feats_fused = (corners_dino_last + corners_siglip_last) * 0.5

        # ---------------------------------------------------------
        # C. 召唤采样器 (现在输入非常纯净且正确)
        # ---------------------------------------------------------
        center_base, selected_patches = self.saliency_sampler(
            center_feat_fused, 
            corner_feats_fused, 
            num_topk=210
        )

        # =========================================================
        # 🧩 阶段三：最终合体 (Final Assembly)
        # =========================================================
        # 顺序：[Global Soul (4)] + [Center Base (144)] + [Corner Gems (210)]
        # 总计：358 Tokens
        
        # final_embedding = torch.cat([
        #     final_global_cls_tokens,   # [B, 4, 1024]
        #     center_base,        # [B, 144, 1024]
        #     selected_patches    # [B, 210, 1024]
        # ], dim=1)

  
        # # =========================================================
        # # 🕵️ RANK 0 深度 Token 审计 (放入 forward 结尾)
        # # =========================================================
        # if rank == 0:
        #     print(f"\n" + "📊" * 10 + " [TOKEN 组成审计] " + "📊" * 10)
        #     print(f"1. 【灵魂层】Global Soul Tokens (Fusion): {final_global_cls_tokens.shape}")
        #     print(f"2. 【骨架层】Center Base (12x12):        {center_base.shape}")
        #     print(f"3. 【血肉层】Selected Corner Patches:   {selected_patches.shape}")
        #     print(f"--------------------------------------------------")
        #     print(f"🚀 [最终合体] Final Embedding Shape:    {final_embedding.shape}")
            
        #     # 关键：检查是否达到了你预期的 365
        #     expected_total = 365
        #     actual_total = final_embedding.shape[1]
        #     if actual_total != expected_total:
        #         print(f"❌ [警报] Token 数量不匹配！预期 {expected_total}, 实际 {actual_total}")
        #         print(f"💡 [提示] 你之前提到的 4+1+144+1+210+5=365 逻辑在代码中尚未体现。")
        #     else:
        #         print(f"✅ [对齐] 365 目标达成！")
        #     print("📊" * 28 + "\n")
      

        B = final_global_cls_tokens.shape[0]
        device = final_global_cls_tokens.device
        dtype = final_global_cls_tokens.dtype
        # 2. 准备分隔符 (从 Parameter 拿到并 expand 到当前 Batch)
        # n_sep1: [B, 1, 1024], n_sep2: [B, 1, 1024], n_end: [B, 5, 1024]
        # 现在 self.n_sep1_embed 存在了，expand 就能跑通了
        n_sep1 = self.n_sep1_embed.expand(B, -1, -1).to(device=device, dtype=dtype)
        n_sep2 = self.n_sep2_embed.expand(B, -1, -1).to(device=device, dtype=dtype)
        n_end  = self.n_end_embed.expand(B, -1, -1).to(device=device, dtype=dtype)

        # 3. 按照 4 + 1 + 144 + 1 + 210 + 5 = 365 焊死
        final_embeddings = torch.cat([
            final_global_cls_tokens,             # 0-3
            n_sep1,           # 4 (Separator 1)
            center_base,      # 5-148
            n_sep2,           # 149 (Separator 2)
            selected_patches, # 150-359
            n_end             # 360-364 (End padding)
        ], dim=1)
        #print(f"🚀 [混合塔返回特征] Final Embedding Shape:    {final_embeddings.shape}")
        return final_embeddings, None