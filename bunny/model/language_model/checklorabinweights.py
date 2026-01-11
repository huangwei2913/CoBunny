import torch
import os

# --- 设置你的路径 ---
bin_path = '/mnt/CoBunny/checkpoints-finetune/bunny-phi1.5-mixed-lora-695k/non_lora_trainables.bin'

def inspect_non_lora_bin(file_path):
    if not os.path.exists(file_path):
        print(f"❌ 找不到文件: {file_path}")
        return

    print(f"🔍 正在读取文件: {file_path}")
    # map_location='cpu' 确保在没有显存的情况下也能读取
    state_dict = torch.load(file_path, map_location='cpu')

    print(f"📊 发现参数总量: {len(state_dict)} 个张量\n")
    print(f"{'参数名称 (Parameter Name)':<60} | {'形状 (Shape)':<20}")
    print("-" * 85)

    for name, tensor in state_dict.items():
        # 打印每一个权重的名称和维度
        print(f"{name:<60} | {str(list(tensor.shape)):<20}")

    # 特别检查：计算一下这部分参数的总量
    total_params = sum(p.numel() for p in state_dict.values())
    print("-" * 85)
    print(f"📈 该文件包含的总参数量: {total_params / 1e6:.2f} M (百万)")

if __name__ == "__main__":
    inspect_non_lora_bin(bin_path)




################
'''看这几行：
🔍 正在读取文件: /mnt/CoBunny/checkpoints-finetune/bunny-phi1.5-mixed-lora-695k/non_lora_trainables.bin
📊 发现参数总量: 119 个张量

参数名称 (Parameter Name)                                        | 形状 (Shape)          
-------------------------------------------------------------------------------------
base_model.model.model.vision_tower.final_cls_weights        | [2]                 
base_model.model.model.vision_tower.dino_cls_attn_weights    | [4]                 
base_model.model.model.vision_tower.oryx_cls_attn_weights    | [4]                 
base_model.model.model.vision_tower.mlp_layers.0.weight      | [1024, 768]         
base_model.model.model.vision_tower.mlp_layers.0.bias        | [1024]              
base_model.model.model.vision_tower.mlp_layers.1.weight      | [1024, 1152]        
base_model.model.model.vision_tower.mlp_layers.1.bias        | [1024]              
base_model.model.model.vision_tower.multi_cls_cross_attn_blocks_A.0.norm1.weight | [1024]              
base_model.model.model.vision_tower.multi_cls_cross_attn_blocks_A.0.norm1.bias | [1024]              
base_model.model.model.vision_tower.multi_cls_cross_attn_blocks_A.0.attn.wq.weight | [1024, 1024]        
base_model.model.model.vision_tower.multi_cls_cross_attn_blocks_A.0.attn.wk.weight | [1024, 1024]        
base_model.model.model.vision_tower.multi_cls_cross_attn_blocks_A.0.attn.wv.weight | [1024, 1024]        
base_model.model.model.vision_tower.multi_cls_cross_attn_blocks_A.0.attn.proj.weight | [1024, 1024]        
base_model.model.model.vision_tower.multi_cls_cross_attn_blocks_A.0.attn.proj.bias | [1024]              
base_model.model.model.vision_tower.multi_cls_cross_attn_blocks_A.0.norm2.weight | [1024]              
base_model.model.model.vision_tower.multi_cls_cross_attn_blocks_A.0.norm2.bias | [1024]              
base_model.model.model.vision_tower.multi_cls_cross_attn_blocks_A.0.mlp.fc1.weight | [4096, 1024]        
base_model.model.model.vision_tower.multi_cls_cross_attn_blocks_A.0.mlp.fc1.bias | [4096]              
base_model.model.model.vision_tower.multi_cls_cross_attn_blocks_A.0.mlp.fc2.weight | [1024, 4096]        
base_model.model.model.vision_tower.multi_cls_cross_attn_blocks_A.0.mlp.fc2.bias | [1024]              
base_model.model.model.vision_tower.multi_cls_cross_attn_blocks_A.1.norm1.weight | [1024]              
base_model.model.model.vision_tower.multi_cls_cross_attn_blocks_A.1.norm1.bias | [1024]              
base_model.model.model.vision_tower.multi_cls_cross_attn_blocks_A.1.attn.wq.weight | [1024, 1024]        
base_model.model.model.vision_tower.multi_cls_cross_attn_blocks_A.1.attn.wk.weight | [1024, 1024]        
base_model.model.model.vision_tower.multi_cls_cross_attn_blocks_A.1.attn.wv.weight | [1024, 1024]        
base_model.model.model.vision_tower.multi_cls_cross_attn_blocks_A.1.attn.proj.weight | [1024, 1024]        
base_model.model.model.vision_tower.multi_cls_cross_attn_blocks_A.1.attn.proj.bias | [1024]              
base_model.model.model.vision_tower.multi_cls_cross_attn_blocks_A.1.norm2.weight | [1024]              
base_model.model.model.vision_tower.multi_cls_cross_attn_blocks_A.1.norm2.bias | [1024]              
base_model.model.model.vision_tower.multi_cls_cross_attn_blocks_A.1.mlp.fc1.weight | [4096, 1024]        
base_model.model.model.vision_tower.multi_cls_cross_attn_blocks_A.1.mlp.fc1.bias | [4096]              
base_model.model.model.vision_tower.multi_cls_cross_attn_blocks_A.1.mlp.fc2.weight | [1024, 4096]        
base_model.model.model.vision_tower.multi_cls_cross_attn_blocks_A.1.mlp.fc2.bias | [1024]              
base_model.model.model.vision_tower.multi_cls_cross_attn_blocks_A.2.norm1.weight | [1024]              
base_model.model.model.vision_tower.multi_cls_cross_attn_blocks_A.2.norm1.bias | [1024]              
base_model.model.model.vision_tower.multi_cls_cross_attn_blocks_A.2.attn.wq.weight | [1024, 1024]        
base_model.model.model.vision_tower.multi_cls_cross_attn_blocks_A.2.attn.wk.weight | [1024, 1024]        
base_model.model.model.vision_tower.multi_cls_cross_attn_blocks_A.2.attn.wv.weight | [1024, 1024]        
base_model.model.model.vision_tower.multi_cls_cross_attn_blocks_A.2.attn.proj.weight | [1024, 1024]        
base_model.model.model.vision_tower.multi_cls_cross_attn_blocks_A.2.attn.proj.bias | [1024]              
base_model.model.model.vision_tower.multi_cls_cross_attn_blocks_A.2.norm2.weight | [1024]              
base_model.model.model.vision_tower.multi_cls_cross_attn_blocks_A.2.norm2.bias | [1024]              
base_model.model.model.vision_tower.multi_cls_cross_attn_blocks_A.2.mlp.fc1.weight | [4096, 1024]        
base_model.model.model.vision_tower.multi_cls_cross_attn_blocks_A.2.mlp.fc1.bias | [4096]              
base_model.model.model.vision_tower.multi_cls_cross_attn_blocks_A.2.mlp.fc2.weight | [1024, 4096]        
base_model.model.model.vision_tower.multi_cls_cross_attn_blocks_A.2.mlp.fc2.bias | [1024]              
base_model.model.model.vision_tower.multi_cls_cross_attn_blocks_A.3.norm1.weight | [1024]              
base_model.model.model.vision_tower.multi_cls_cross_attn_blocks_A.3.norm1.bias | [1024]              
base_model.model.model.vision_tower.multi_cls_cross_attn_blocks_A.3.attn.wq.weight | [1024, 1024]        
base_model.model.model.vision_tower.multi_cls_cross_attn_blocks_A.3.attn.wk.weight | [1024, 1024]        
base_model.model.model.vision_tower.multi_cls_cross_attn_blocks_A.3.attn.wv.weight | [1024, 1024]        
base_model.model.model.vision_tower.multi_cls_cross_attn_blocks_A.3.attn.proj.weight | [1024, 1024]        
base_model.model.model.vision_tower.multi_cls_cross_attn_blocks_A.3.attn.proj.bias | [1024]              
base_model.model.model.vision_tower.multi_cls_cross_attn_blocks_A.3.norm2.weight | [1024]              
base_model.model.model.vision_tower.multi_cls_cross_attn_blocks_A.3.norm2.bias | [1024]              
base_model.model.model.vision_tower.multi_cls_cross_attn_blocks_A.3.mlp.fc1.weight | [4096, 1024]        
base_model.model.model.vision_tower.multi_cls_cross_attn_blocks_A.3.mlp.fc1.bias | [4096]              
base_model.model.model.vision_tower.multi_cls_cross_attn_blocks_A.3.mlp.fc2.weight | [1024, 4096]        
base_model.model.model.vision_tower.multi_cls_cross_attn_blocks_A.3.mlp.fc2.bias | [1024]              
base_model.model.model.vision_tower.b_pseudo_cls_head.score_predictor.0.weight | [2048, 1024]        
base_model.model.model.vision_tower.b_pseudo_cls_head.score_predictor.0.bias | [2048]              
base_model.model.model.vision_tower.b_pseudo_cls_head.score_predictor.2.weight | [1, 2048]           
base_model.model.model.vision_tower.b_pseudo_cls_head.score_predictor.2.bias | [1]                 
base_model.model.model.vision_tower.multi_cls_cross_attn_blocks_B.0.norm1.weight | [1024]              
base_model.model.model.vision_tower.multi_cls_cross_attn_blocks_B.0.norm1.bias | [1024]              
base_model.model.model.vision_tower.multi_cls_cross_attn_blocks_B.0.attn.wq.weight | [1024, 1024]        
base_model.model.model.vision_tower.multi_cls_cross_attn_blocks_B.0.attn.wk.weight | [1024, 1024]        
base_model.model.model.vision_tower.multi_cls_cross_attn_blocks_B.0.attn.wv.weight | [1024, 1024]        
base_model.model.model.vision_tower.multi_cls_cross_attn_blocks_B.0.attn.proj.weight | [1024, 1024]        
base_model.model.model.vision_tower.multi_cls_cross_attn_blocks_B.0.attn.proj.bias | [1024]              
base_model.model.model.vision_tower.multi_cls_cross_attn_blocks_B.0.norm2.weight | [1024]              
base_model.model.model.vision_tower.multi_cls_cross_attn_blocks_B.0.norm2.bias | [1024]              
base_model.model.model.vision_tower.multi_cls_cross_attn_blocks_B.0.mlp.fc1.weight | [4096, 1024]        
base_model.model.model.vision_tower.multi_cls_cross_attn_blocks_B.0.mlp.fc1.bias | [4096]              
base_model.model.model.vision_tower.multi_cls_cross_attn_blocks_B.0.mlp.fc2.weight | [1024, 4096]        
base_model.model.model.vision_tower.multi_cls_cross_attn_blocks_B.0.mlp.fc2.bias | [1024]              
base_model.model.model.vision_tower.multi_cls_cross_attn_blocks_B.1.norm1.weight | [1024]              
base_model.model.model.vision_tower.multi_cls_cross_attn_blocks_B.1.norm1.bias | [1024]              
base_model.model.model.vision_tower.multi_cls_cross_attn_blocks_B.1.attn.wq.weight | [1024, 1024]        
base_model.model.model.vision_tower.multi_cls_cross_attn_blocks_B.1.attn.wk.weight | [1024, 1024]        
base_model.model.model.vision_tower.multi_cls_cross_attn_blocks_B.1.attn.wv.weight | [1024, 1024]        
base_model.model.model.vision_tower.multi_cls_cross_attn_blocks_B.1.attn.proj.weight | [1024, 1024]        
base_model.model.model.vision_tower.multi_cls_cross_attn_blocks_B.1.attn.proj.bias | [1024]              
base_model.model.model.vision_tower.multi_cls_cross_attn_blocks_B.1.norm2.weight | [1024]              
base_model.model.model.vision_tower.multi_cls_cross_attn_blocks_B.1.norm2.bias | [1024]              
base_model.model.model.vision_tower.multi_cls_cross_attn_blocks_B.1.mlp.fc1.weight | [4096, 1024]        
base_model.model.model.vision_tower.multi_cls_cross_attn_blocks_B.1.mlp.fc1.bias | [4096]              
base_model.model.model.vision_tower.multi_cls_cross_attn_blocks_B.1.mlp.fc2.weight | [1024, 4096]        
base_model.model.model.vision_tower.multi_cls_cross_attn_blocks_B.1.mlp.fc2.bias | [1024]              
base_model.model.model.vision_tower.multi_cls_cross_attn_blocks_B.2.norm1.weight | [1024]              
base_model.model.model.vision_tower.multi_cls_cross_attn_blocks_B.2.norm1.bias | [1024]              
base_model.model.model.vision_tower.multi_cls_cross_attn_blocks_B.2.attn.wq.weight | [1024, 1024]        
base_model.model.model.vision_tower.multi_cls_cross_attn_blocks_B.2.attn.wk.weight | [1024, 1024]        
base_model.model.model.vision_tower.multi_cls_cross_attn_blocks_B.2.attn.wv.weight | [1024, 1024]        
base_model.model.model.vision_tower.multi_cls_cross_attn_blocks_B.2.attn.proj.weight | [1024, 1024]        
base_model.model.model.vision_tower.multi_cls_cross_attn_blocks_B.2.attn.proj.bias | [1024]              
base_model.model.model.vision_tower.multi_cls_cross_attn_blocks_B.2.norm2.weight | [1024]              
base_model.model.model.vision_tower.multi_cls_cross_attn_blocks_B.2.norm2.bias | [1024]              
base_model.model.model.vision_tower.multi_cls_cross_attn_blocks_B.2.mlp.fc1.weight | [4096, 1024]        
base_model.model.model.vision_tower.multi_cls_cross_attn_blocks_B.2.mlp.fc1.bias | [4096]              
base_model.model.model.vision_tower.multi_cls_cross_attn_blocks_B.2.mlp.fc2.weight | [1024, 4096]        
base_model.model.model.vision_tower.multi_cls_cross_attn_blocks_B.2.mlp.fc2.bias | [1024]              
base_model.model.model.vision_tower.multi_cls_cross_attn_blocks_B.3.norm1.weight | [1024]              
base_model.model.model.vision_tower.multi_cls_cross_attn_blocks_B.3.norm1.bias | [1024]              
base_model.model.model.vision_tower.multi_cls_cross_attn_blocks_B.3.attn.wq.weight | [1024, 1024]        
base_model.model.model.vision_tower.multi_cls_cross_attn_blocks_B.3.attn.wk.weight | [1024, 1024]        
base_model.model.model.vision_tower.multi_cls_cross_attn_blocks_B.3.attn.wv.weight | [1024, 1024]        
base_model.model.model.vision_tower.multi_cls_cross_attn_blocks_B.3.attn.proj.weight | [1024, 1024]        
base_model.model.model.vision_tower.multi_cls_cross_attn_blocks_B.3.attn.proj.bias | [1024]              
base_model.model.model.vision_tower.multi_cls_cross_attn_blocks_B.3.norm2.weight | [1024]              
base_model.model.model.vision_tower.multi_cls_cross_attn_blocks_B.3.norm2.bias | [1024]              
base_model.model.model.vision_tower.multi_cls_cross_attn_blocks_B.3.mlp.fc1.weight | [4096, 1024]        
base_model.model.model.vision_tower.multi_cls_cross_attn_blocks_B.3.mlp.fc1.bias | [4096]              
base_model.model.model.vision_tower.multi_cls_cross_attn_blocks_B.3.mlp.fc2.weight | [1024, 4096]        
base_model.model.model.vision_tower.multi_cls_cross_attn_blocks_B.3.mlp.fc2.bias | [1024]              
base_model.model.model.mm_projector.0.weight                 | [2048, 1024]        
base_model.model.model.mm_projector.0.bias                   | [2048]              
base_model.model.model.mm_projector.2.weight                 | [2048, 2048]        
base_model.model.model.mm_projector.2.bias                   | [2048]              
-------------------------------------------------------------------------------------
📈 该文件包含的总参数量: 111.11 M (百万)

...vision_tower.mlp_layers.0.weight: [1024, 768] (可能是处理 DINOv2 768维特征)

...vision_tower.mlp_layers.1.weight: [1024, 1152] (可能是处理 SigLIP/Oryx 1152维特征)

...vision_tower.multi_cls_cross_attn_blocks_A/B: 这是最值钱的地方！你的模型并不是简单的把两张图拼接，而是引入了 Cross-Attention（交叉注意力机制）。

Block A 和 Block B：说明模型在视觉端就在做深度的特征融合。

Score Predictor：说明模型甚至有“注意力筛选”能力，会自动判断哪些视觉特征更重要。

结论： 这是一个比普通 Bunny 模型高级得多的**“动态特征融合视觉塔”**。

2. 视觉到语言的“翻译官” (Projector)
看最后几行：

mm_projector.0.weight: [2048, 1024]

mm_projector.2.weight: [2048, 2048] 这里清晰地显示：视觉端出来的特征是 1024 维，经过一个两层的 MLP 映射到了 Phi-1.5 的 2048 维 语言空间。

3. 为什么你必须进行合并 (Merge)？
注意看参数的 Key（键名）： 它们全部是以 base_model.model.model.... 开头的。
'''
################