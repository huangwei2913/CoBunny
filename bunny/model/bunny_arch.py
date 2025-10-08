from abc import ABC, abstractmethod

import torch

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector

from bunny.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX


#直接强制选择这个indeity投影，直接在这个里面，在增加一个开源的openvision2的编码器，并初始化装载
class BunnyMetaModel:

    def __init__(self, config):
        super(BunnyMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=not getattr(config, 'continuous_training', False))
            if getattr(config, 'continuous_training', False):
                config.continuous_training = False
            self.mm_projector = build_vision_projector(config)

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args):
        vision_tower = model_args.vision_tower
        #这个地方传递进了模型的视觉塔名称，我们要在这里加入
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter #这个其实就是预训练的适配器

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)
            self.vision_tower = vision_tower
        else:
            vision_tower = self.vision_tower
            vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type')
        self.config.mm_hidden_size = vision_tower.hidden_size #投影层的输入特征维度要等于视觉编码器的特征维度

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')

            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))


class BunnyMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        image_features = self.get_model().mm_projector(image_features)
        #
        return image_features

    # 也就说这个地方可以加入一个专家混合（MOE）层，这个层负责更加合理地融合来自多个视觉编码器的不同视觉特征
    def prepare_inputs_labels_for_multimodal(
            self, input_ids, position_ids, attention_mask, past_key_values, labels, images
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if past_key_values is not None and vision_tower is not None and images is not None and input_ids.shape[
                1] == 1:
                target_shape = past_key_values[-1][-1].shape[-2] + 1
                attention_mask = torch.cat((attention_mask, torch.ones(
                    (attention_mask.shape[0], target_shape - attention_mask.shape[1]),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                )), dim=1)
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if type(images) is list or images.ndim == 5:
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            image_features = [x.flatten(0, 1).to(self.device) for x in image_features]
        else:
            image_features = self.encode_images(images).to(self.device)

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        input_ids_temp = input_ids # points to the actual input_ids tensor

        # remove the padding using attention_mask -- TODO: double check
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in
                     zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        # -- TODO: better implementation?
        # replace IMAGE_TOKEN_INDEX(-200) with 0 to be compatible with repetition penalty
        input_ids_temp[input_ids_temp == IMAGE_TOKEN_INDEX] = 0

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [
                cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1:image_token_indices[i + 1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i] + 1:image_token_indices[i + 1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(
                        torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device,
                                   dtype=cur_labels.dtype))

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype,
                                       device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype,
                                                              device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype,
                                                             device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels



# 这段 prepare_inputs_labels_for_multimodal 方法的核心作用是：

# 主要功能
# 将视觉特征和文本token的输入融合成语言模型可接受的统一输入格式。

# 为什么要计算图像特征？
# 图片（视觉信息）和文本是多模态模型的两个输入来源。

# 视觉部分需要经过视觉编码器，转成与文本token相似的嵌入向量（token embedding），才能和文本token放到一起送到语言模型。

# 这里的“计算影像特征”不是简单的“占位符编码”，而是用视觉编码器把图像编码成有意义的视觉特征向量，这些向量表示图像内容，和文本token嵌入在同一个向量空间或相近空间。

# 视觉特征是融合后的视觉表示，语言模型decoder才能利用这些信息执行理解或生成任务。

# 如何融合？
# 代码中先对图片batch中的每张图片用视觉编码器编码成向量序列；

# 再根据配置对视觉token做变形（flatten，reshape，unpad等），保证视觉token维度适合序列拼接；

# 将视觉token按照位置插入到对应文本token序列中（文本中用特殊token标记视觉token位置），替换原来文本输入中对应的视觉占位符；

# 这样视觉向量和文本token统一形成输入，可以送入语言模型。

# 总结
# 这段代码的关键目的是把图像转成语言模型能理解的向量token，并把这些token和文本token序列融合在一起，为多模态语言模型的推理和训练做准备。

# 所以这里计算影像特征，是把图片的低级像素信息经过视觉编码器转成高级语义特征，作为多模态模型理解视觉内容的“语言描述”，而不是简单占用位置的“占位符”。您的理解如果是视觉token替代文本中的占位符，是对的，但视觉token本身是由视觉编码器编码的有效语义表示，不是空占位符。

# 如需，可详细分步解释每段代码处理过程。简单总结，prepare_inputs_labels_for_multimodal 是多模态输入预处理的桥梁，实现文本和视觉内容在序列级的融合。这段代码中的计算影像特征并不是简单的对图片做占位符编码，而是通过视觉编码器将图片编码成对语言模型有意义的视觉特征向量。具体来说：

# 输入的图片先经过视觉编码器，编码成一系列视觉token（向量序列），这些向量是图像的高级语义特征，类似文本token的词嵌入。

# 这些视觉token会经过处理（如展平、变形、unpad），确保形状和序列结构适合与文本token统一拼接。

# 把处理好的视觉token插入到文本token序列中指定的位置，替代原先文本中表示图像位置的占位符token。

# 最终形成一个包含文本信息和视觉信息混合的token序列，作为多模态语言模型的输入。

# 所以，计算影像特征是让图片信息转换成语言模型可以理解和使用的语义向量，而不是简单占位符。这是多模态大语言模型融合视觉和文本信息的基础步骤。您的理解“图片和文字作为输入时，对输入的图片进行编码成为LLM decoder需要的tokens”基本正确，只是要理解视觉tokens本身含有图像的语义信息，
# 不是空占位符。


# BunnyMetaForCausalLM 是一个多模态语言模型的基类接口，负责管理：

# 视觉编码器和多模态投影模块的调用；

# 多模态输入的预处理（文本token + 视觉特征嵌入）；

# 和训练/推理过程中输入张量和标签的对齐与组织。

# 这是Bunny多模态大模型体系中连接视觉编码器与语言模型的核心模块，实现了视觉与文本信息的统一输入准备，非常关键于多模态任务的高效训练和推理。

# 简单形象的说，它就是多模态语言模型里的“多模态输入融合器”，桥接视觉和语言两大模态，让它们协同工作，是你整体多模态模型架构不可或缺的核心部分。这个 BunnyMetaForCausalLM 类是一个抽象基类，主要用来管理多模态因果语言模型中的视觉编码器和视觉投影模块的调用，以及多模态输入（文本token与视觉特征嵌入）的融合处理。它的核心职责包括：

# 通过抽象方法 get_model() 访问底层模型实例；

# 调用视觉编码器（vision tower）和视觉投影器（mm_projector）负责将图像转换为视觉特征向量；

# 在 prepare_inputs_labels_for_multimodal 方法中，将视觉特征合理插入到文本token嵌入序列中，处理多图像或多维视觉数据，并完成attention mask、位置编码和labels的对齐；

# 支持多模态输入并进行序列长度调整、填充和截断等预处理；

# 代码注释中提到可在此模块中加入专家混合（MoE）层，实现多个视觉编码器特征的融合，从而提升多模态表达能力。

# 简而言之，BunnyMetaForCausalLM 类是 Bunny 多模态大语言模型中的“多模态输入融合器”，在视觉特征编码和语言模型输入之间架起桥梁，确保视觉与文本信息能够同步、高效地被模型接受和处理，是整个多模态模型训练和推理过程中非常关键的模块。

# 方法 encode_images 中，先通过视觉编码器（vision tower）对输入的图像进行编码，获得视觉特征向量，然后通过 mm_projector 模块对视觉特征进行线性变换或适配。

# 在 prepare_inputs_labels_for_multimodal 方法中，融合了视觉特征和文本的输入嵌入：

# 先处理传入的文本 token id 和视觉图像，编码得到视觉特征。

# 对输入序列中的特殊视觉 token（用 IMAGE_TOKEN_INDEX 标记的）进行定位，将对应的位置替换成视觉特征的向量表示。

# 将文本的嵌入向量和视觉嵌入根据位置拼接起来，形成一个结合了多模态信息的新的输入嵌入序列。

# 还会根据 padding 方向对新融合的输入嵌入序列做动态填充，使得每个 batch 输入长度一致，方便后续批处理训练。

# tokenizer_model_max_length：

# 这是模型配置里保存的最大序列长度，限制模型输入（token序列）的最大长度。

# 在代码中，tokenizer_model_max_length用来截断融合了图像嵌入和文本嵌入的新输入序列，确保序列不会超出模型支持的最大长度，避免超出模型设计输入范围。

# 换句话说，就是告诉模型输入最长不能超过多少token或嵌入向量。

# input_ids 和 labels 的区别：

# input_ids 是模型的输入序列的token id列表，代表的是输入文本（或含视觉token）的token编码。

# labels 是训练时模型用于计算损失的目标值（ground truth）。在语言模型训练中，通常labels是input_ids的一个偏移版本（比如下一个token的预测目标）。

# labels中用特殊数值（如IGNORE_INDEX）标记的部分表示不参与loss计算，比如视觉token对应位置。

# 简单说，input_ids是模型看到的输入，labels是模型学习去预测的目标输出。

# 总结：

# tokenizer_model_max_length控制输入序列的最大长度，防止输入过长超出模型限制。

# input_ids是输入的token序列，labels是训练时对应的预测目标，两者功能不同但一一对应。

# 这种设计对于多模态（图像+文本）输入的融合与训练尤为重要，因为视觉token与文本token一起构成了完整序列，模型既看见输入，也有对应的训练目标