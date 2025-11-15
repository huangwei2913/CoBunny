from abc import ABC, abstractmethod
import torch
from .multimodal_encoder.builder import build_vision_tower
from .multimodal_resampler.builder import build_vision_resampler
from .multimodal_projector.builder import build_vision_projector
from bunny.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX
#直接强制选择这个indeity投影，直接在这个里面，在增加一个开源的openvision2的编码器，并初始化装载
class BunnyMetaModel:
    def __init__(self, config):
        super(BunnyMetaModel, self).__init__(config)
        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=not getattr(config, 'continuous_training', False))
            #这个地方可以添加的，因为不传递的话，就是indentifymap
            self.vision_resampler = build_vision_resampler(config, vision_tower=self.vision_tower)     
            if getattr(config, 'continuous_training', False):
                config.continuous_training = False
            self.mm_projector = build_vision_projector(config)
    #注意这里写法，其实不是获取命令行中的字符串
    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)  #这只是从self对象拿到vision_tower属性（模型对象），如果是list则取第一个，否则原样返回
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower
    def initialize_vision_modules(self, model_args):
        vision_tower = model_args.vision_tower
        #这个地方传递进了模型的视觉塔名称，我们要在这里加入
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter #这个其实就是预训练的适配器
        self.config.mm_vision_tower = vision_tower
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)
            vision_resampler = build_vision_resampler(model_args, vision_tower=vision_tower)
            for k, v in vision_resampler.config.items():
                setattr(self.config, k, v)
            self.vision_tower = vision_tower            #给自身的vision_tower对象属性赋值
            self.vision_resampler = vision_resampler           
        else:
            vision_tower = self.vision_tower
            vision_resampler = self.vision_resampler
            vision_tower.load_model()
            for p in self.vision_resampler.parameters():
                p.requires_grad = True
        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type')
        self.config.mm_resampler_type = getattr(model_args, 'mm_resampler_type')
        if self.config.mm_resampler_type=='masked_drop':
            self.config.mm_hidden_size = vision_tower.hidden_size #投影层的输入特征维度要等于视觉编码器的特征维度
        elif self.config.mm_resampler_type=='spatial_pool':
            self.config.mm_hidden_size = vision_resampler.out_channels #投影层的输入特征维度要等于视觉编码器的特征维度
        elif self.config.mm_resampler_type=='qformer':
            self.config.mm_hidden_size = vision_resampler.hidden_size #投影层的输入特征维度要等于视觉编码器的特征维度
        elif self.config.mm_resampler_type=='dynamic_compressor':
            self.config.mm_hidden_size = vision_resampler.hidden_size #投影层的输入特征维度要等于视觉编码器的特征维度
        else:
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
            incompatible_keys = self.vision_resampler.load_state_dict(get_w(mm_projector_weights, 'vision_resampler'), strict=False)
            print(incompatible_keys)

class BunnyMetaForCausalLM(ABC):
    @abstractmethod
    def get_model(self):
        pass
    def get_vision_tower(self):
        return self.get_model().get_vision_tower()
    def encode_images(self, images):
        #这里可以来控制,如果不是dynamic 
        mm_resampler_type = getattr(self.config, 'mm_resampler_type', None)
        if mm_resampler_type is None:  # 常规处理模式, 这里我们希望的
            image_features, _ = self.get_model().get_vision_tower()(images)  #这里是希望能返回中间层特征
            image_features = self.get_model().mm_projector(image_features)
            return image_features
        else:  #如果是那几个
            if mm_resampler_type=='dynamic_compressor':
                image_features, image_size, _ = self.get_model().get_vision_tower()(images)
                image_features,_ = self.get_model().vision_resampler(image_features, forward_type='image',image_size=image_size)
                image_features = self.get_model().mm_projector(image_features)
                return image_features
            else:
                image_features = self.get_model().get_vision_tower()(images)
                image_features = self.get_model().vision_resampler(image_features)
                image_features = self.get_model().mm_projector(image_features)
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
        #images 是一个包含多张图像的 List (每个元素是 $B \times C \times H \times W$)
        #5D 张量 (视频格式 $B \times T \times C \times H \times W$)，则判断为多帧输入 
        if type(images) is list or images.ndim == 5:
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images) 
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            #编码完成后，它将返回的特征拆分回原来每张图像的特征 List  
            image_features = [x.flatten(0, 1).to(self.device) for x in image_features]
        else:
            image_features  = self.encode_images(images).to(self.device)   #这个地方可能是不需要的
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
        #这个地方有一个隐含的错误，如果Token indices sequence length is longer than the specified maximum sequence length 
        # for this model (3052 > 2048). Running this sequence through the model will result in indexing errors（我们在最后面再来修改*************）
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
