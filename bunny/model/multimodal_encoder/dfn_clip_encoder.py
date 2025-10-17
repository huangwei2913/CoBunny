import torch
from open_clip import create_model_from_pretrained 

from .base_encoder import ProcessorWrapper
from .clip_encoder_dfn import ClipVisionTower_


class DfnClipVisionTower(ClipVisionTower_):
    def load_model(self, device_map=None):
        local_weight_path = '/home/huangwei/DFN5B-CLIP-ViT-H-14-378/open_clip_pytorch_model.bin'
        if self.vision_tower_name == "apple/DFN5B-CLIP-ViT-H-14-378":
            #clip_model, processor = create_model_from_pretrained('hf-hub:apple/DFN5B-CLIP-ViT-H-14-384')
            clip_model, processor = create_model_from_pretrained(
                model_name='ViT-H-14',
                pretrained=local_weight_path
            )
        elif self.vision_tower_name == "apple/DFN2B-CLIP-ViT-L-14":
            clip_model, processor = create_model_from_pretrained('hf-hub:apple/DFN2B-CLIP-ViT-L-14')
        else:
            raise ValueError(f'Unknown vision tower: {self.vision_tower_name}')
        
        self.vision_tower = clip_model.visual
        self.vision_tower.output_tokens = True  #这个开关变量很重要
        # 要让模型输出所有patch tokens的表示（而不仅仅是最后的全局特征向量），可以借助视觉编码器中的方法或参数控制。在许多 ViT（Vision Transformer）的实现中，模型默认只返回CLS Token或池化后的全局特征，但若想输出所有patch token的特征，通常需要：
        # 设置模型某个属性，如output_tokens=True（你之前提到过），告诉模型返回全部token。
        # 直接调用视觉编码器的底层接口获取全部tokens
        self._hidden_size = clip_model.visual.ln_post.normalized_shape[0] #patch embeddings→Transformer layers→LayerNorm (ln_post)→cls token作为全局特征输出
        self._image_size = clip_model.visual.image_size[0]  #从标量属性中获得
        self._patch_size = clip_model.visual.patch_size[0]  #从标量属性中获得
        self.image_processor = ProcessorWrapper(processor, height=self._image_size, width=self._image_size)
        self.vision_tower.requires_grad_(self.unfreeze_mm_vision_tower)
        self.is_loaded = True

    def _forward(self, images):
        with torch.set_grad_enabled(self.unfreeze_mm_vision_tower):
            _, image_features = self.vision_tower(images.to(device=self.device, dtype=self.dtype))
            return image_features



# 你问到：

# python
# self._hidden_size = clip_model.visual.ln_post.normalized_shape[0]
# self._image_size = clip_model.visual.image_size[0]
# self._patch_size = clip_model.visual.patch_size[0]
# 这些属性为什么可以直接访问，但模型结构里没有明显显示normalized_shape、image_size或patch_size字段。

# 原因分析
# ln_post 是一个 LayerNorm 层

# ln_post 是模型视觉编码器（VisionTransformer）最后一个层归一化层（LayerNorm），PyTorch的 LayerNorm 对象。

# normalized_shape 是 LayerNorm 层的定义属性，代表其归一化的维度大小（这里是1280），这是PyTorch官方API的标准属性，所以能访问到。

# 例如 clip_model.visual.ln_post.normalized_shape 会返回一个tuple (1280,)，代表输出的隐藏特征维度。

# image_size 和 patch_size

# 这两个是VisionTransformer实现中的自定义属性，通常在ViT模型初始化时设置。

# 具体来说，这是为了方便代码读取模型接受的输入图像尺寸（如224）和分割成patch的大小（如14）。

# 这两个属性一般是在继承VisionTransformer的类或模型配置中通过self.image_size、self.patch_size保存在对象属性里的。

# 它们可能不会像网络层那样细节暴露在print(model)的结构中，但是在模型类实例中是可以访问的。

# 你为什么没有看到它们？
# 如果你只看print(model)或print(model.visual)，这些字段可能不会作为网络层被显式打印，因为它们是标量属性而非nn.Module层。

# 你可以用print(vars(clip_model.visual))查看visual对象包含的所有属性。通常可以看到image_size和patch_size等。

# 这属于模型实现细节，属于Python类的普通属性，不是模型的子模块，所以不会出现在网络层结构列表中。

# 你可以试试的调试代码
# python
# print("LayerNorm normalized shape:", clip_model.visual.ln_post.normalized_shape)
# print("Vision tower image size:", getattr(clip_model.visual, "image_size", "No attribute"))
# print("Vision tower patch size:", getattr(clip_model.visual, "patch_size", "No attribute"))

# # 查看 visual 对象的所有普通属性
# print(vars(clip_model.visual))
# 总结
# ln_post.normalized_shape 是PyTorch LayerNorm的官方属性，代表归一化维度。

# image_size、patch_size是自定义属性，虽然属于模型实例，但因为非nn.Module层所以不显示在层结构中。

# 你需要用vars()或getattr()等方法直接访问Python对象属性查看这些值。

# 这样解释，你就清楚为什么打印模型结构看不到它们，但可以直接访问到这些属性了。这些属性中，ln_post是PyTorch中LayerNorm层的一个实例，
# 它本身有一个官方定义的属性normalized_shape，代表归一化维度大小，所以你能通过clip_model.visual.ln_post.normalized_shape直接访问。
# 而image_size和patch_size通常是VisionTransformer类中特意设置的模型配置属性，虽然它们不是网络层（nn.Module），因此在打印模型结构时不会显示，
# 但它们是该模型实例的普通对象属性，可以直接访问。

# 如果你想查看这些属性，可以这样做：

# python
# print("LayerNorm normalized shape:", clip_model.visual.ln_post.normalized_shape)
# print("Vision tower image size:", getattr(clip_model.visual, "image_size", "No attribute"))
# print("Vision tower patch size:", getattr(clip_model.visual, "patch_size", "No attribute"))

# # 查看 visual 对象所有属性
# print(vars(clip_model.visual))
# 总结：

# normalized_shape是PyTorch LayerNorm的标准属性。

# image_size和patch_size是模型类自带的普通属性，不属于网络层，所以打印结构时不会显示。

# 你要用vars()或getattr()查看这些Python对象的普通属性，而不仅是print(model)的网络层结构。