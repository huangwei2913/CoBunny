import re
import math
from torch import nn
from functools import partial
from timm.layers.norm_act import LayerNormAct2d
from torchvision.ops.misc import SqueezeExcitation as SELayer
from torchvision.models.mobilenetv3 import InvertedResidual, InvertedResidualConfig


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class Minigpt(nn.Module):
    def __init__(self, config=None):
        super(Minigpt, self).__init__()
        # c*4 is the input size, and c is the output size for the linear layer
        inc, ouc = config.mm_hidden_size, config.hidden_size
        self.linear = nn.Linear(inc * 4, ouc)

    def forward(self, x):
        # x is the input tensor with shape [b, num_tokens, c]
        b, num_tokens, c = x.shape

        # Check if num_tokens is divisible by 4
        if num_tokens % 4 != 0:
            raise ValueError("num_tokens must be divisible by 4")

        # Reshape x to [b, num_tokens/4, c*4]
        x = x.view(b, num_tokens // 4, c * 4)

        # Apply the linear transformation
        x = self.linear(x)
        return x


class Vanilla(nn.Module):
    def __init__(self, config=None):
        super(Vanilla, self).__init__()
        # c*4 is the input size, and c is the output size for the linear layer
        inc, ouc = config.mm_hidden_size, config.hidden_size
        self.linear = nn.Linear(inc * 4, ouc)

    def forward(self, x):
        b, num_tokens, c = x.shape

        # Check if num_tokens is divisible by 4
        if num_tokens % 4 != 0:
            raise ValueError("num_tokens must be divisible by 4")

        # First, reshape to [b, num_tokens//4, 4, c]
        x = x.view(b, num_tokens // 4, 4, c)

        # Then, permute to interleave the tokens
        x = x.permute(0, 1, 3, 2).contiguous()

        # Finally, reshape to [b, num_tokens//4, c*4] to interleave features of 4 tokens
        x = x.view(b, num_tokens // 4, c * 4)

        # Apply the linear transformation
        x = self.linear(x)
        return x


class LDPBlock(nn.Module):
    # Lightweight Downsample Projector Block

    def __init__(self, config=None):
        super().__init__()

        inc, ouc = config.mm_hidden_size, config.hidden_size
        layer_norm = partial(LayerNormAct2d, act_layer=None)
        se_layer = partial(SELayer, scale_activation=nn.Hardsigmoid)
        self.mlp = nn.Sequential(
            nn.Identity(), nn.Linear(inc, ouc), nn.GELU(), nn.Linear(ouc, ouc)
        )
        self.mb_block = nn.Sequential(
            nn.Identity(),
            InvertedResidual(InvertedResidualConfig(ouc, 3, ouc, ouc, True, "HS", 1, 1, 1), layer_norm, se_layer),
            InvertedResidual(InvertedResidualConfig(ouc, 3, ouc, ouc, True, "HS", 2, 1, 1), layer_norm, se_layer)
        )

    def forward(self, x):
        b, num_tokens, c = x.shape
        h = int(math.sqrt(num_tokens))
        x = self.mlp(x)
        x = x.permute(0, 2, 1).reshape(b, -1, h, h)
        x = self.mb_block(x)
        x = x.flatten(2).permute(0, 2, 1)
        return x


class LDPNetProjector(nn.Module):

    def __init__(self, config=None):
        super().__init__()
        self.model = LDPBlock(config)

    def forward(self, x):
        return self.model(x)


class SPP(nn.Module):

    def __init__(self, config=None, projector_type='v1'):
        super().__init__()

        self.projector_type = projector_type

        inc, ouc = config.mm_hidden_size, config.hidden_size
        self.linear_0 = nn.Linear(inc, inc)

        self.linear_1 = nn.Linear(inc, ouc)

        self.pooling = nn.AvgPool2d(kernel_size=2)

        self.linear_2 = nn.Linear(ouc, ouc)

    def forward(self, x):
        b, num_tokens, c = x.shape
        h = int(math.sqrt(num_tokens))
        if 'v1' in self.projector_type:
            x = self.linear_1(x)
            x = x.permute(0, 2, 1).reshape(b, -1, h, h)
            x = self.pooling(x)
            x = x.flatten(2).permute(0, 2, 1)
            x = self.linear_2(x)
        elif 'v2' in self.projector_type:
            x = self.linear_1(x)
            x = self.linear_2(x)
            x = x.permute(0, 2, 1).reshape(b, -1, h, h)
            x = self.pooling(x)
            x = x.flatten(2).permute(0, 2, 1)
        elif 'v3' in self.projector_type:
            x = self.linear_0(x)
            x = x.permute(0, 2, 1).reshape(b, -1, h, h)
            x = self.pooling(x)
            x = x.flatten(2).permute(0, 2, 1)
            x = self.linear_1(x)
            x = self.linear_2(x)
        return x


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'mlp2x_gelu')

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    elif projector_type.startswith('mlp'):
        mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
        if mlp_gelu_match:
            mlp_depth = int(mlp_gelu_match.group(1))
            modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(config.hidden_size, config.hidden_size))
            return nn.Sequential(*modules)

    elif projector_type.startswith('spp'):
        return SPP(config, projector_type)

    elif projector_type == 'ldp':
        return LDPNetProjector(config)

    elif projector_type == 'vanilla':
        return Vanilla(config)

    elif projector_type == 'minigpt':
        return Minigpt(config)

    elif projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')




# 1. IdentityMap 类
# 这是一个空的恒等映射模块，forward直接返回输入x不做任何改变。

# 常用于需要占位或跳过某个处理步骤的时候，比如条件性地替代其他模块。

# config属性返回配置信息，表明这是一个“identity”类型的投影器。

# 2. Minigpt 类
# 输入是形状 [batch_size, num_tokens, c] 的张量。

# 其设计思路是：将num_tokens维度上的token数量四个为一组合并，也就是说reshape为 [batch_size, num_tokens/4, c*4]。

# 再经过一个线性层把c*4维转回hidden_size（ouc）。

# 调用了view重塑张量。注意要求num_tokens必须是4的倍数。

# 主要功能是“压缩token数量”，将4个token的特征并到一个token上。

# 3. Vanilla 类
# 结构类似Minigpt，但对token做了交织（interleave）变换：

# 先reshape成 [batch_size, num_tokens/4, 4, c]

# permute维度顺序，实现4个token特征的交错

# 再reshape成 [batch_size, num_tokens/4, c*4]

# 其目的也是合并4个token特征，但顺序混合，可能提升特征的表达力。

# 4. LDPBlock 类（Lightweight Downsample Projector Block）
# 这是一个轻量级下采样投影块，包含了：

# 一个多层感知机(MLP)，包括两个线性层和GELU激活。

# 一个MobileNetInvertedResidual模块序列，使用两个残差块，带有SE（Squeeze-Excitation）注意力机制。

# 这部分代码用来对token特征做降维和空间结构上的下采样。

# 它会将[batch, tokens, channels]的输入视为二维空间，并将token展平转换为二维图像块后做卷积处理。

# 5. LDPNetProjector 类
# 简单封装LDPBlock，作为更高层的投影器模块。

# 6. SPP 类（Spatial Pyramid Pooling）
# 这是一个空间金字塔池化模块变体，支持三种版本（v1, v2, v3），用不同方式对token特征做线性变换、池化和非线性转换。

# 输入x形状 [batch, tokens, channels]，先变换后转为[batch, channels, h, h]，做AvgPool2d后再变回原结构。

# 目的是通过多层次空间池化精炼特征，减少token数量同时保留更多上下文信息。

# 7. 函数 build_vision_projector
# 根据config.mm_projector_type参数动态创建相应的“投影器”模块。它根据类型返回不同的模块实例，支持线性、mlp（多层感知机）、SPP、LDP、Vanilla、Minigpt、Identity等。

# 这种动态构建方式方便统一接口调用不同特征变换和降维方法。

# 总结
# 整体上，这组代码是针对视觉-语言大模型（如多模态学习模型）中视觉特征后处理设计的“投影器”模块，负责：

# 聚合多个token特征（如Minigpt、Vanilla）

# 对token空间特征做下采样和通道混合（如LDPBlock, SPP）

# 提供不同复杂度投影器实现以适应不同需求和计算预算

# IdentityMap方便快捷替代

# 这些模块能帮助模型在保持视觉特征丰富性的同时，实现更高效的特征压缩与表达。

# 这样模块化设计方便在训练和微调阶段试验不同投影策略，选出效果最佳方案，提升下游任务性能。下面逐个简要解释代码中各个模块的作用：

# IdentityMap
# 恒等映射，forward直接返回输入不变。

# 用作占位或者跳过某个变换模块。

# config表明它代表“identity”类型。

# Minigpt
# 作用是把输入张量中num_tokens维度4个token合并成1个，进行线性映射。

# 要求num_tokens是4的倍数，先reshape为[b, num_tokens/4, c*4]后线性变换。

# 适合对token数量做降维和合并。

# Vanilla
# 类似Minigpt，但在合并4个token特征前做了交织（permute）操作，以增强特征表达力。

# 输入先reshape到[b, num_tokens/4, 4, c]，permute再reshape到[b, num_tokens/4, c*4]。

# LDPBlock（轻量级下采样投影块）
# 含MLP网络和两个MobileNet InvertedResidual模块，带SE注意力。

# 用于下采样和更复杂的空间通道特征变换，将token展成二维特征图处理。

# LDPNetProjector
# LDPBlock的封装，用作投影器模块接口。

# SPP（空间金字塔池化变体）
# 三个版本的池化加线性层模块，先映射通道，再reshape为二维特征图池化，最后映射输出。

# 用于多层次空间池化特征表达。

# build_vision_projector函数
# 根据配置中mm_projector_type动态选择和返回以上各种投影器模块实例。

# 支持线性、mlp多层感知机、spp、ldp、vanilla、minigpt、identity等投影方式。

# 总结来说，这些模块都是针对视觉特征的不同方式的“投影器”，负责：

# 聚合或合并token特征（Minigpt、Vanilla）

# 进行空间下采样和特征变换（LDPBlock、SPP）

# 提供多样的设计方案灵活选择

# IdentityMap为占位或无变换模块

# 用于多模态模型或视觉编码器中，对视觉token特征做高效压缩、表达和空间处理，提升模型性能和效率。