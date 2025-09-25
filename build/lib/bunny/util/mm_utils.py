import base64
import torch

from PIL import Image
from io import BytesIO
from transformers import StoppingCriteria

from bunny.constants import IMAGE_TOKEN_INDEX


def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))

#将原来的图片中的大的边为准，来构建一个正方形图像，将原图拷贝到新建立的这个正方形图像上
def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result



# image_aspect_ratio 是取模型配置model_cfg里名为 image_aspect_ratio 的属性，如果没有则为None。

# 如果该属性是 'pad'，说明需要把图片处理为正方形，即保持较大边长，较小边长进行填充（补全）不拉伸。

# 这时程序会对每张图片调用你之前见过的expand2square函数，把图片置入一个正方形画布，空白处用图片均值（image_processor.image_mean）乘以255转成RGB背景色填充。

# 然后使用image_processor.preprocess将图片转换成模型需要的格式（比如归一化、调整大小、转Tensor等），并提取pixel_values，最后保留第一个batch维度（因为单张处理）。

# 如果image_aspect_ratio不是'pad'，则直接调用image_processor按默认方式批量处理图像，通常包含resize、裁剪但不负责填充正方形。

# 最后判断处理后所有图片尺寸是否一致，如果一致，把图片列表用torch.stack合并成一批，方便后续模型批量输入。

def process_images(images, image_processor, model_cfg):
    image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
    new_images = []
    if image_aspect_ratio == 'pad':
        for image in images:
            image = expand2square(image, tuple(int(x * 255) for x in image_processor.image_mean))
            image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            new_images.append(image)
    else:
        return image_processor(images, return_tensors='pt')['pixel_values']
    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = torch.stack(new_images, dim=0)
    return new_images





#insert_separator result: [[101, 10, 20], [-200], [30, 40], [-200]]
#final input_ids: [101, 10, 20, -200, 30, 40]
#只要知道是在序列中加入image，让image的token进行穿插，并考虑第一个分段的其实是不是bos
#IMAGE_TOKEN_INDEX = -200 首先，以<image>分割输入文本，得到多段文本，然后对每一段tokenizer化，得到多段文本对应的tokens序列，也就是分段tokens
def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]  #以<image>为界，把字符串分成多段文本（去掉图片标记的位置）

    #在 X 的每两个元素之间插入一个分隔符 sep[i]，但不在最后一个元素后再加一个分隔符。
    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

    #然后，检查首个分段tokens中的第一个token是
    input_ids = []
    offset = 0
    #是看prompt_chunks的第一段token序列第一个token，是不是正好是代表序列开始的特殊token。如果是包含的，这说明是开始
    #于是用 offset = 1 记录这个“偏移”，并把这个BOS token先放到input_ids列表开头
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1 #如果是的话，说明当前这段文本编码里带有句子起始符（BOS），因为有了这个符号，后面拼接token用的时候有偏移了，所以要做特殊处理。
        input_ids.append(prompt_chunks[0][0]) #于是用 offset = 1 记录这个“偏移”，并把这个BOS token先放到input_ids列表开头

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])   #简单理解就是“丢弃碎片开头第一位BOS，除非是第一个token，避免重复开始标记”。

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids

# tokenizer_image_token 函数的主要作用是对包含 <image> 标记的文本序列，进行分段token化，并用特殊图片tokenID替代 <image> 的位置，返回模型可以识别的token id序列。
#假定promots是"<image>\nWhat is this?" 那其实就是将<image>对应位置用对应的图像token来填充

def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        self.max_keyword_len = 0
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            if len(cur_keyword_ids) > 1 and cur_keyword_ids[0] == tokenizer.bos_token_id:
                cur_keyword_ids = cur_keyword_ids[1:]
            if len(cur_keyword_ids) > self.max_keyword_len:
                self.max_keyword_len = len(cur_keyword_ids)
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]

    def call_for_batch(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        offset = min(output_ids.shape[1] - self.start_len, self.max_keyword_len)
        self.keyword_ids = [keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids]
        for keyword_id in self.keyword_ids:
            truncated_output_ids = output_ids[0, -keyword_id.shape[0]:]
            if torch.equal(truncated_output_ids, keyword_id):
                return True
        outputs = self.tokenizer.batch_decode(output_ids[:, -offset:], skip_special_tokens=True)[0]
        for keyword in self.keywords:
            if keyword in outputs:
                return True
        return False

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        outputs = []
        for i in range(output_ids.shape[0]):
            outputs.append(self.call_for_batch(output_ids[i].unsqueeze(0), scores))
        return all(outputs)
