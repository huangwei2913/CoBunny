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
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()
        self.is_loaded = False
        self.args = args
        self.global_dimension = getattr(args, "mm_hidden_size", 1024)
        self.compression_K = getattr(args, "compression_K", 8)
        self.num_heads = 8 
        self.mlp_ratio = 4.0
        self.target_image_size = 378
        self.image_processor = ImageProcessorMultipleEncoders()
        self.unfreeze_mm_vision_tower = getattr(args, 'unfreeze_mm_vision_tower', False)
        self.args = args

        self.dino_vision_tower = DinoVisionTower(args.vision_tower_dino, args, delay_load=delay_load)
        self.siglip_vision_tower = SiglipVisionTower(args.vision_tower_siglip, args, delay_load=delay_load)
        self.mlp_layers = nn.ModuleList([
            nn.Linear(self.dino_vision_tower.hidden_size, self.global_dimension),
            nn.Linear(self.siglip_vision_tower.hidden_size, self.global_dimension)
        ])

        self._init_interaction_modules()
        if not delay_load: 
            self.load_model()


    def _set_subtower_grad_state(self):
        """统一管理子塔的梯度和模式状态"""
        for sub_tower in [self.siglip_vision_tower, self.dino_vision_tower]:
            if sub_tower is not None:
                # 🚨 修复关键：判断是否有 config 属性再赋值
                if hasattr(sub_tower, 'config'):
                    sub_tower.config.unfreeze_mm_vision_tower = self.unfreeze_mm_vision_tower
                
                # 既然 DINO 没有 config，我们就直接把属性挂在它对象上，
                # 这样以后 check 属性时也能找到，且不会报错
                sub_tower.unfreeze_mm_vision_tower = self.unfreeze_mm_vision_tower
                
                if self.unfreeze_mm_vision_tower:
                    sub_tower.requires_grad_(True)
                    sub_tower.train()
                else:
                    sub_tower.requires_grad_(False)
                    sub_tower.eval()


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

        def check_tower_valid(tower, attr_name):
            try:
                # 检查子塔是否已经具备了 Backbone 实例
                if hasattr(tower, attr_name):
                    param = next(tower.parameters())
                    # 只有当权重不是随机初始化的（std=1.0）且有数值时才有效
                    return param.numel() > 0 and torch.std(param.data).item() != 1.0
            except:
                return False
            return False
        

        # 1. 尝试探测当前内存中的权重是否为“有效”权重
        siglip_valid = check_tower_valid(self.siglip_vision_tower, 'vision_tower')
        dino_valid = check_tower_valid(self.dino_vision_tower, 'vision_tower')
        
        is_weight_valid = siglip_valid and dino_valid
        

        # 2. 身份识别与自动回退
        if self.is_loaded or is_weight_valid:
            # 情况 A: 权重已经正确加载（来自 Checkpoint）
            print("🚀 [Weight Verified] 检测到有效权重，跳过官方路径加载。")
            self._set_subtower_grad_state()
            self.is_loaded = True
            return
        else:
            # 情况 B: 权重是随机的（Checkpoint 没喂进去或 Key 没对上）
            # 这种情况我们要强制执行官方加载，确保模型不崩
            if is_weight_valid == False and getattr(self.args, 'model_name_or_path', None):
                print("⚠️ [Warning] Checkpoint 中未发现有效的视觉塔权重（可能 Key 不匹配）！")
                print("🛠️ [Fallback] 正在回退至官方初始权重以确保训练/推理正确...")
            
            print(f"🏗️ 加载 DINO 官方权重: {self.args.vision_tower_dino}")
            self.dino_vision_tower.load_model()
            print(f"🏗️ 加载 SigLIP 官方权重: {self.args.vision_tower_siglip}")
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
        # =================================================================
        # 1. 维度归一化：处理 Tensor、List 以及各种套娃维度
        # =================================================================
        if isinstance(images, dict): 
            images = images.get("pixel_values", images)
        
        # 🚨 变量初始化，防止任何条件分支下的 NameError
        is_multi_view = False
        Num_Views = 6  
        n_enc = 2      # 双塔 (DINO + SigLIP)
        c, h, w = 3, 378, 378 # 强制对齐 378 物理维度

        # 处理 DataCollator 堆叠失败返回的 List 情况
        if isinstance(images, list):
            # 只要子图都是 378，这里 cat 就能把 [2,6...] 和 [1,6...] 拼成 [3,6...]
            images = torch.cat(images, dim=0) 
        
        if not isinstance(images, torch.Tensor):
            images = self.image_processor.preprocess(images, return_tensors='pt')["pixel_values"]

        images = images.to(device=self.device, dtype=self.dtype)

        # 核心逻辑：利用 numel 暴力推算总图数，管它是几维，统一摊平
        # 这样无论输入是 [B, N, 6, 2, 3, 378, 378] 还是 [Total_B_Views, 2, 3, 378, 378] 都能处理
        Total_B = images.numel() // (n_enc * c * h * w)
        
        # 💡 自动判断是否需要执行 6->1 视图融合
        # 逻辑：如果图片总数能被 6 整除，且大于 0，我们假设它是 6 视图模式
        if Total_B >= Num_Views and Total_B % Num_Views == 0:
            is_multi_view = True

        # 强制平铺为 [总单图数, 2, 3, 378, 378]
        images = images.view(Total_B, n_enc, c, h, w)

        # 索引解包：0 给 DINO，1 给 SigLIP
        dino_input = images[:, 0].contiguous()
        siglip_input = images[:, 1].contiguous()

        # =================================================================
        # 2. 特征提取与多层拆分
        # =================================================================
        _, A_inter = self.dino_vision_tower(dino_input)
        _, B_inter = self.siglip_vision_tower(siglip_input)
        
        A_proj = self.mlp_layers[0](A_inter.to(self.dtype))
        del A_inter
        B_proj = self.mlp_layers[1](B_inter.to(self.dtype)) 
        del B_inter
        # 这里的 1000 是个阈值，用来区分“单层特征”还是“多层拼接特征”
        if self.N_layer_A > 1: 
            L_per_layer = A_proj.shape[1] // self.N_layer_A
            A_full = A_proj.view(Total_B, self.N_layer_A, L_per_layer, -1)
            B_full = B_proj.view(Total_B, self.N_layer_B, L_per_layer, -1)
            
            A_cls, A_patches = A_full[:, :, 0:1, :], A_full[:, :, 1:, :] 
            B_patches = B_full[:, :, 1:, :]
            T_actual = L_per_layer - 1 # 减去 CLS 后的 patch 数
        else:
            # 兜底：处理单层或非标准特征
            A_cls = A_proj.mean(dim=1, keepdim=True).unsqueeze(1)
            A_patches = A_proj.unsqueeze(1)
            B_patches = B_proj.unsqueeze(1)
            T_actual = A_proj.shape[1]

        # 锁定当前处理的层数，防止后续 index 越界
        N_A = self.N_layer_A if A_proj.shape[1] > 1000 else 1
        N_B = self.N_layer_B if A_proj.shape[1] > 1000 else 1

        # =================================================================
        # 3. 压缩与交互 (Cross-Attention)
        # =================================================================
        target_context = T_actual // self.compression_K
        
        def get_kv_ctx(p):
            p_flat = p.flatten(0, 1).contiguous()
            m = bipartite_soft_matching_merge(p_flat, target_context, p_flat)
            return m.view(Total_B, -1, self.global_dimension)

        B_kv_context = get_kv_ctx(B_patches)
        A_kv_context = get_kv_ctx(A_patches)

        def cross_attn_group(cls_t, kv_t, blocks, weights):
            enhanced = []
            for i in range(len(blocks)):
                if i < cls_t.shape[1]:
                    # CLS 特征与对侧 KV 特征交互
                    enhanced.append(blocks[i](torch.cat([cls_t[:, i], kv_t], dim=1))[:, 0:1, :])
            if not enhanced: return cls_t[:, 0]
            w = F.softmax(weights[:len(enhanced)], dim=0).view(1, -1, 1).to(self.dtype)
            return torch.sum(torch.cat(enhanced, dim=1) * w, dim=1, keepdim=True)

        # 执行双塔交互逻辑
        final_cls_A = cross_attn_group(A_cls, B_kv_context, self.multi_cls_cross_attn_blocks_A, self.dino_cls_attn_weights)
        
        # 伪 CLS 头的权重预测
        pseudo_B = self.b_pseudo_cls_head(B_kv_context.unsqueeze(1).repeat(1, N_B, 1, 1).flatten(0, 1)).view(Total_B, N_B, 1, -1)
        final_cls_B = cross_attn_group(pseudo_B, A_kv_context, self.multi_cls_cross_attn_blocks_B, self.oryx_cls_attn_weights)

        # 融合两者的最终 CLS Token
        w_final = F.softmax(self.final_cls_weights, dim=0).view(1, 2, 1).to(self.dtype)
        enhanced_cls_token = torch.sum(torch.cat([final_cls_A, final_cls_B], dim=1) * w_final, dim=1, keepdim=True)

        # =================================================================
        # 4. 子图内部合并 (Intra-View Merge) -> 输出 365 Tokens
        # =================================================================
        A_up, A_lo = A_patches[:, :N_A//2].flatten(1, 2), A_patches[:, N_A//2:].flatten(1, 2)
        B_up, B_lo = B_patches[:, :N_B//2].flatten(1, 2), B_patches[:, N_B//2:].flatten(1, 2)

        def merge_to(x, n):
            if x.shape[1] <= n: return x 
            m_f, _ = random_bipartite_soft_matching(x, r=x.shape[1]-n)
            return m_f(x)

        target_n = T_actual // self.compression_K 
        
        # 拼接产生单图的 365 个 Token
        out = torch.cat([enhanced_cls_token, merge_to(A_up, target_n), merge_to(B_up, target_n), merge_to(A_lo, target_n), merge_to(B_lo, target_n)], dim=1)
        
        # 如果不是多视图（比如单图测试），直接返回
        if not is_multi_view:
            return out, None

        # =================================================================
        # 5. 6视图最终聚合 (6 -> 1)
        # =================================================================
        # 恢复 Batch 结构 [Batch, 6, 365, 1024]
        Real_Batch_Size = Total_B // Num_Views
        all_tokens = out.view(Real_Batch_Size, Num_Views, out.shape[1], -1).flatten(1, 2)
        
        # 核心：使用 Token Merging 算法将 2190 个 Token 压缩回 365 个
        # 这里的 365 必须与 LLM 端定义的长度严格一致
        final_out = bipartite_soft_matching_merge(all_tokens, 365, all_tokens)
        
        return final_out, None