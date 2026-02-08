import dataclasses
from enum import auto, Enum
from typing import List, Tuple, Union
import base64
from io import BytesIO
from PIL import Image

# 必须确保你的 constants.py 中定义了 DEFAULT_IMAGE_TOKEN = "<img_content>"
from bunny.constants import DEFAULT_IMAGE_TOKEN

class SeparatorStyle(Enum):
    """
    定义对话的分隔符风格。
    TWO: 用于类似 User/Assistant 的多轮对话 (Phi-1.5/Vicuna)
    PLAIN: 用于简单的单轮指令或纯文本补全
    """
    TWO = auto()
    PLAIN = auto()

@dataclasses.dataclass
class Conversation:
    """
    对话管理类：负责将 JSON 中的对话列表拼接到一个大的 String 中。
    核心升级：在此处进行 Token 替换和多图锚点注入。
    """
    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle
    sep: str = "###"
    sep2: str = None
    version: str = "Unknown"

    skip_next: bool = False

    def get_prompt(self):
        """
        核心方法：生成最终输入给模型的文本。
        在此处解决 ID 污染和多图逻辑。
        """
        messages = self.messages
        processed_messages = []  # 只针对第一个阶段训练有效 
        for role, msg in messages: # 只针对第一个阶段训练有效 
            if msg is not None:
                if isinstance(msg, tuple):
                    msg, _, _ = msg
                # 调用你定义的 process_image_tokens
                msg = self.process_image_tokens_stage1(msg)
                processed_messages.append([role, msg])
            else:
                processed_messages.append([role, None])  # 只针对第一个阶段训练有效 

        # 1. 处理 PLAIN 模式 (通常用于简单指令)
        if self.sep_style == SeparatorStyle.PLAIN:
            ret = ""
            if self.system:
                ret += self.system + self.sep
            
            for i, (role, msg) in enumerate(processed_messages):
                if msg:
                    # 直接拼内容，完全不加 "USER:" 或 "ASSISTANT:"
                    ret += msg + self.sep
                else:
                    ret += ""
            return ret    

        # 2. 处理 TWO 模式 (Phi-1.5 训练主要走这里)
        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(messages):
                if message:
                    if isinstance(message, tuple):
                        message = message[0]
                    
                    # 【核心修复 2】: 调用处理函数，解决双图和 Token 问题
                    msg_content = self.process_image_tokens(message)

                    ret += role + ": " + msg_content + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    #第一个阶段的目标只是训练projector，所有合格阶段
    def process_image_tokens_stage1(self, text: str) -> str:
        RAW_IMG_TAG = "<image>" 
        if RAW_IMG_TAG in text:
            num_images = text.count(RAW_IMG_TAG)
            if num_images == 1:
                # [微调]：确保 <img_content> 独占一行或有明显分隔，有利于 Stage 1 学习
                return text.replace(RAW_IMG_TAG, DEFAULT_IMAGE_TOKEN + "\n")
            
            elif num_images > 1:
                parts = text.split(RAW_IMG_TAG)
                new_text = ""
                for i in range(num_images):
                    # 你的多图锚点逻辑很棒，保留
                    new_text += f"{parts[i]}Image {i+1}: {DEFAULT_IMAGE_TOKEN} "
                new_text += parts[-1]
                return new_text.strip()
        return text

    #后面的第二和第三个阶段使用bunny做对话模版才用这个          
    def process_image_tokens(self, text: str) -> str:
        """
        [关键逻辑]
        1. 解决污染：将 "<image>" 替换为 DEFAULT_IMAGE_TOKEN (<img_content>)
        2. 解决幻觉：如果是多图，自动注入 "Image 1:", "Image 2:" 锚点
        """
        # 原始数据集里的占位符
        RAW_IMG_TAG = "<image>" 
        
        if RAW_IMG_TAG in text:
            # 统计图片数量
            num_images = text.count(RAW_IMG_TAG)
            
            if num_images == 1:
                # 单图情况：直接替换名字，避开 50256 污染
                return text.replace(RAW_IMG_TAG, DEFAULT_IMAGE_TOKEN)
            
            elif num_images > 1:
                # 多图情况：不仅换名字，还要加坐标
                parts = text.split(RAW_IMG_TAG)
                new_text = ""
                for i in range(num_images):
                    # 注入逻辑：Image 1: <img_content>
                    # 这里的空格很重要，帮助分词器切分
                    new_text += f"{parts[i]}Image {i+1}: {DEFAULT_IMAGE_TOKEN} "
                new_text += parts[-1]
                return new_text.strip()
            
        return text

    def append_message(self, role, message):
        self.messages.append([role, message])

    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            version=self.version)

    def to_gradio_chatbot(self):
        # 保持你原有的 Gradio 显示逻辑不变
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    msg, image, image_process_mode = msg
                    max_hw, min_hw = max(image.size), min(image.size)
                    aspect_ratio = max_hw / min_hw
                    max_len, min_len = 800, 400
                    shortest_edge = int(min(max_len / aspect_ratio, min_len, min_hw))
                    longest_edge = int(shortest_edge * aspect_ratio)
                    W, H = image.size
                    if H > W:
                        H, W = longest_edge, shortest_edge
                    else:
                        H, W = shortest_edge, longest_edge
                    image = image.resize((W, H))
                    buffered = BytesIO()
                    image.save(buffered, format="JPEG")
                    img_b64_str = base64.b64encode(buffered.getvalue()).decode()
                    img_str = f'<img src="data:image/png;base64,{img_b64_str}" alt="user upload image" />'
                    msg = img_str + msg.replace('<image>', '').strip()
                    ret.append([msg, None])
                else:
                    ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret
    
    # 保留原有的 dict 方法和 get_images 方法...
    def dict(self):
         return {
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
        }

# --- 预设模板实例 (保持不变) ---
conv_bunny = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the user's questions.",
    roles=("USER", "ASSISTANT"),
    version="bunny",
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="<|endoftext|>",
)
#增加第一个阶段的对话模版
conv_plain = Conversation(
    system="",
    roles=("", ""), # 角色为空
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.PLAIN,
    sep="\n",       # Stage 1 通常每一段后面只跟一个换行
    sep2=None,
    version="plain"
)

default_conversation = conv_bunny
conv_templates = {
    "default": conv_bunny,
    "bunny": conv_bunny,
    "plain": conv_plain,
}