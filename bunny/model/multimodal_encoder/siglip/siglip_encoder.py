import torch
import torch.nn as nn
from transformers import SiglipVisionModel, SiglipImageProcessor, SiglipVisionConfig
from functools import partial
import torch.nn.functional as F
import math
from einops import rearrange

# split_chessboard 保持不变
def split_chessboard(x, num_split):
    # ... (保持不变)
    B, C, H, W = x.shape
    assert H % num_split == 0 and W % num_split == 0
    h, w = H // num_split, W // num_split
    x_split = torch.cat([x[:, :, i*h:(i+1)*h, j*w:(j+1)*w] for i in range(num_split) for j in range(num_split)], dim=0)
    return x_split

# batched_forward 保持不变
def batched_forward(model, x, batch_size=-1):
    if batch_size == -1:
        return model(x)
    else:
        x_batched = x.split(batch_size)
        outs = [model(x) for x in x_batched]
        return torch.cat(outs, dim=0)

# *** 关键修改：merge_chessboard_multilayers ***
# 目的：能够对一个张量列表（即多层特征）中的每一层进行合并
def merge_chessboard_multilayers(x_list, num_split):
    """
        x_list: List[torch.Tensor], 包含所有隐藏层的张量列表。
        对列表中的每个张量执行 merge_chessboard 逆向操作。
    """
    merged_list = []
    
    # 原始的单层合并逻辑 (现在作为一个内部函数)
    def _merge_single_tensor(x, num_split):
        B, C, H, W = x.shape
        assert B % (num_split**2) == 0
        b = B // (num_split**2)
        x_merge = torch.cat([torch.cat([x[(i*num_split + j)*b:(i*num_split + j + 1)*b] for j in range(num_split)], dim=-1)
                            for i in range(num_split)], dim=-2)
        return x_merge

    # 对每一层特征进行合并
    for x in x_list:
        merged_list.append(_merge_single_tensor(x, num_split))
        
    return merged_list # 返回一个包含所有合并后特征的列表


# --- multiscale_forward 必须大幅修改以适应多层特征处理 ---
# *** 关键修改：multiscale_forward_multilayers ***
def multiscale_forward(model, input, scales=None, img_sizes=None, max_split_size=None, resize_output_to_idx=0, num_prefix_token=0,
             output_shape='bnc', split_forward=False, selected_layers_indices=None):
    """
    修改后的 multiscale_forward：
    1. model 必须返回所有隐藏状态张量列表。
    2. 只对 selected_layers_indices 中指定的层进行棋盘合并、插值和拼接等后处理。
    """
    assert input.dim() == 4, "Input image must be in the shape of BxCxHxW."
    assert input.shape[2] == input.shape[3], "Currently only square images are supported."
    assert output_shape in ['bnc', 'bchw'], "Output shape should be either BxNxC (e.g., ViT) or BxCxHxW (e.g., ConvNet)."
    assert output_shape == 'bnc' or num_prefix_token == 0, "For ConvNet there shouldn't be any prefix token."

    b, c, input_size, _ = input.shape

    # 1. 准备多尺度输入 (S 尺度)
    img_sizes = img_sizes or [int(input_size * scale) for scale in scales]
    max_split_size = max_split_size or input_size
    num_splits = [math.ceil(size / max_split_size) for size in img_sizes]
    input_multiscale = []
    for size, num_split in zip(img_sizes, num_splits):
        x = F.interpolate(input.to(torch.float32), size=size, mode='bicubic').to(input.dtype)
        x = split_chessboard(x, num_split=num_split) # 假设 split_chessboard 可用
        input_multiscale.append(x)

    # 2. 运行前向传播：只运行一次视觉塔 (S 个尺度分片组成的超大批次)
    # outs_multiscale_layers 的结构：[S, L] (S 个尺度，每个包含 L 个层的特征列表)
    outs_multiscale_layers = []
    for x in input_multiscale: # 遍历 S 个尺度
        outs_layers = batched_forward(model, x, b) if split_forward else model(x) # 假设 batched_forward 可用
        outs_multiscale_layers.append(outs_layers)

    # 3. 后处理和合并
    
    # 3.1. 层过滤逻辑
    L = len(outs_multiscale_layers[0])
    processed_indices = []
    if selected_layers_indices is None:
        selected_layers_indices = list(range(L))
    
    for idx in selected_layers_indices:
        if idx < 0:
            idx = L + idx # 处理负索引
        if 0 <= idx < L:
            processed_indices.append(idx)

    # 创建过滤后的结构：[L_filtered, S]
    layers_multiscale = []
    for l_idx in processed_indices:
        # layers_multiscale[l] 是第 l 层的 S 个尺度分片结果列表
        layers_multiscale.append([outs_multiscale_layers[s][l_idx] for s in range(len(img_sizes))])
    
    L_filtered = len(layers_multiscale)
    if L_filtered == 0:
        return [] # 没有层被选中

    # 3.2. 前缀 Token (CLS/Patch Embed Token) 移除与重排 (针对 L_filtered 层)
    
    # 预先提取前缀 Token，以便在最终拼接时使用
    outs_prefix_multiscale_layers = None
    if num_prefix_token > 0:
        # 结构：[L_filtered, S]
        outs_prefix_multiscale_layers = [[out[:, :num_prefix_token] for out in outs_scale] 
                                         for outs_scale in layers_multiscale]
        # 移除特征中的前缀 Token
        layers_multiscale = [[out[:, num_prefix_token:] for out in outs_scale] 
                             for outs_scale in layers_multiscale]

    if output_shape == 'bnc':
        # 将 BxNxC 转换为 BxCxHxW (方便 merge 和 interpolate)
        new_layers_multiscale = []
        for outs_scale in layers_multiscale: # 遍历 L_filtered 层
            new_outs_scale = []
            for out in outs_scale: # 遍历 S 个尺度（分块后的结果）
                N = out.shape[1]; C = out.shape[2]
                H = W = int(N ** 0.5)
                new_outs_scale.append(rearrange(out, 'b (h w) c -> b c h w', h=H, w=W))
            new_layers_multiscale.append(new_outs_scale)
        layers_multiscale = new_layers_multiscale # 结构仍为 [L_filtered, S]

    # 3.3. 转置回 [S, L_filtered] 结构，以匹配 num_splits 进行合并
    # 结构：[S, L_filtered]
    merged_by_scale_layers = [[layers_multiscale[l][s] for l in range(L_filtered)] 
                              for s in range(len(img_sizes))]

    # 3.4. 合并棋盘分片 (对 L_filtered 层独立操作)
    merged_layers_multiscale = []
    for outs_scale_all_layers, num_split in zip(merged_by_scale_layers, num_splits):
        # outs_scale_all_layers 是 List[Tensor] (L_filtered 层在当前 S 尺度上的分块结果)
        # 假设 merge_chessboard_multilayers 可用
        merged_layers = merge_chessboard_multilayers(outs_scale_all_layers, num_split=num_split) 
        merged_layers_multiscale.append(merged_layers) # 结构：[S, L_filtered]

    # 3.5. 再次转置结构：[S, L_filtered] 转为 [L_filtered, S]，以便按层进行插值
    merged_layers_multiscale_transposed = [[merged_layers_multiscale[s][l] for s in range(len(img_sizes))] 
                                           for l in range(L_filtered)]

    # 3.6. 尺度插值和拼接 (对 L_filtered 层独立操作)
    final_output_layers = []
    output_size = merged_layers_multiscale_transposed[0][resize_output_to_idx].shape[-2]
    
    for outs_multiscale in merged_layers_multiscale_transposed: # 遍历 L_filtered 层
        out_layer = torch.cat([F.interpolate(outs_multiscale[i].to(torch.float32), size=output_size,
                                             mode='area').to(outs_multiscale[i].dtype)
                               for i in range(len(outs_multiscale))], dim=1)
        final_output_layers.append(out_layer)
        
    # 3.7. 最终 Rearrange (BxCxHxW -> BxNxC)
    if output_shape == 'bnc':
        final_output_layers = [rearrange(out_layer, 'b c h w -> b (h w) c') for out_layer in final_output_layers]

    # 4. 前缀 Token 最终处理 (应用于 L_filtered 层)
    if num_prefix_token > 0:
        final_with_prefix = []
        # outs_prefix_multiscale_layers 结构：[L_filtered, S]
        for l_idx in range(L_filtered):
            outs_prefix_multiscale_single_layer = outs_prefix_multiscale_layers[l_idx] # S 个尺度的前缀 Token 列表
            
            outs_prefix_multiscale_single_layer_averaged = []
            for s in range(len(img_sizes)):
                out_prefix = outs_prefix_multiscale_single_layer[s]
                # 对所有分片结果取平均，还原回 [B, num_prefix, C] 
                averaged_prefix = torch.stack(out_prefix.split(b, dim=0), dim=0).mean(dim=0)
                outs_prefix_multiscale_single_layer_averaged.append(averaged_prefix)
            
            # 将 S 个尺度的前缀 Token 沿着 C 维度拼接
            out_prefix_multiscale = torch.cat(outs_prefix_multiscale_single_layer_averaged, dim=-1)
            
            # 拼接最终特征 [B, N_prefix*S + N_patch*S, C]
            out = torch.cat([out_prefix_multiscale, final_output_layers[l_idx]], dim=1)
            final_with_prefix.append(out)
            
        return final_with_prefix

    return final_output_layers



class SiglipVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()
        self.is_loaded = False
        self.vision_tower_name = vision_tower
        self.select_layer = -1
        self.layer_size_ = None  #隐藏层层数
        self.layer_embed_dim  = None
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
    def feature_select(self, image_forward_outs, layer= None):
        if layer is None:
            image_features = image_forward_outs.hidden_states[self.select_layer]
        else:
           image_features = image_forward_outs.hidden_states[layer]     
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
        for i, h in enumerate(image_forward_outs.hidden_states):
            print(f"Image forward hidden_states[{i}] shape: {tuple(h.shape)}")
        if self.layer_size_ is None:
            self.layer_size_ = len(image_forward_outs.hidden_states)
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
    @property
    def layer_size(self):
        return self.layer_size_  #总共有多少层
    
class SiglipVisionTowerS2(SiglipVisionTower):
    def __init__(self, vision_tower, args, delay_load=False):
        self.s2_scales = getattr(args, 's2_scales', '384,768,1152')
        self.s2_scales = list(map(int, self.s2_scales.split(',')))
        self.s2_scales.sort()
        self.s2_split_size = self.s2_scales[0]
        self.s2_image_size = self.s2_scales[-1]
        self._layer_size_  = None # 层大小
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

    def forward_feature(self, images, selected_layer=None):
        """
        这个函数被 multiscale_forward 调用，它现在必须返回所有隐藏状态的列表。
        selected_layer 参数在这里不再使用，因为我们总是返回所有层。
        """
        image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype),
                                               output_hidden_states=True)
        
        if self._layer_size_ is None:
            self._layer_size_ = len(image_forward_outs.hidden_states)

        # 返回一个张量列表，这是修改后的 multiscale_forward 所期望的
        # 注意：这里我们只返回 Patch Tokens (假设 SiglipVisionModel 的隐藏状态列表第一个是 CLS/Patch Embed Token)
        # 根据 SiglipVisionModel 的输出，outs.hidden_states 包含 [CLS_Token + Patch_Tokens]。
        # 我们假设 multiscale_forward 的 rearrange 逻辑会处理 CLS Token。
        return image_forward_outs.hidden_states
    
    # *** 核心修改：新的 forward 方法，只需一次多尺度计算 ***
    def forward(self, images):

        is_list = type(images) is list
        
        # 1. 统一输入为批次张量 (优化步骤保持不变)
        if is_list:
            images = torch.stack(images)
            
        
        layers_to_process = [0, 1, -1]  #在这里定义选择多少层
        # 2. 运行一次多尺度前向计算，获取所有层的合并特征列表
        # outs_all_layers: List[Tensor], 其中 outs_all_layers[l] 是第 l 层的最终特征张量 [B, N_patches * N_scales, C]
        outs_all_layers = self.multiscale_forward(self.forward_feature, images,
                                                   img_sizes=self.s2_scales,
                                                 max_split_size=self.s2_split_size,
                                                 selected_layers_indices=layers_to_process
                                                 )
        
        # 3. 提取所需的输出
        
        # image_features: 最后一层 (默认层) 的特征
        # 最后一层通常是 outs_all_layers[-1]
        image_features = outs_all_layers[-1]

        # patch_tokens_gallery: 前两层 (第 0 和 第 1 层) 的特征，并拼接在一起
        # outs_all_layers[0] 是第 0 层特征，outs_all_layers[1] 是第 1 层特征
        patch_tokens_gallery = [outs_all_layers[0], outs_all_layers[1]]#在这里可以进行修改

        tensor_list = [t.contiguous() for t in patch_tokens_gallery]
        # 结果 out 的形状将是 [B, 2 * N_patches * N_scales, C]
        out = torch.cat(tensor_list, dim=1)
        
        # 4. 返回结果
        return image_features, out

    @property
    def hidden_size(self):
        return self.config.hidden_size * len(self.s2_scales)


    @property
    def patch_size(self):
        return self.config.patch_size
		