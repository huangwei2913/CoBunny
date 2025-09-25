import dataclasses
from enum import auto, Enum
from typing import List


class SeparatorStyle(Enum):
    """Different separator style."""
    TWO = auto()
    PLAIN = auto()


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
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
        messages = self.messages
        if len(messages) > 0 and type(messages[0][1]) is tuple:
            messages = self.messages.copy()
            init_role, init_msg = messages[0].copy()
            init_msg = init_msg[0].replace("<image>", "").strip()
            if 'mmtag' in self.version:
                messages[0] = (init_role, init_msg)
                messages.insert(0, (self.roles[0], "<Image><image></Image>"))
                messages.insert(1, (self.roles[1], "Received."))
            else:
                messages[0] = (init_role, "<image>\n" + init_msg)

        if self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"

        elif self.sep_style == SeparatorStyle.PLAIN:
            seps = [self.sep, self.sep2]
            ret = self.system
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += message + seps[i % 2]
                else:
                    ret += ""
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

        return ret

    def append_message(self, role, message):
        self.messages.append([role, message])

# get_images(self, return_pil=False):所以对话里面是可以包含影像的，消息结构不仅支持文本，也支持带图像的多模态联合消息，这样方便多模态模型使用文本和图像一起作为输入。这个代码说明对话消息self.messages中，消息内容msg不仅可能是普通字符串文本，也可能是一个元组，元组内含有文本、PIL图片对象和图片处理模式。

# 简单说，消息里可以“夹带”图像：

# 当msg是tuple时，它包含这几个元素：文本描述、PIL图片对象、图片处理方式；

# 该函数遍历这些消息，提取其中的图片，根据指定方式（填充成正方形、裁剪、resize等）处理；

# 最后根据参数决定返回PIL图片对象或转成base64字符串，方便显示。



    def get_images(self, return_pil=False):
        images = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    import base64
                    from io import BytesIO
                    from PIL import Image
                    msg, image, image_process_mode = msg
                    if image_process_mode == "Pad":
                        def expand2square(pil_img, background_color=(122, 116, 104)):
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

                        image = expand2square(image)
                    elif image_process_mode in ["Default", "Crop"]:
                        pass
                    elif image_process_mode == "Resize":
                        image = image.resize((336, 336))
                    else:
                        raise ValueError(f"Invalid image_process_mode: {image_process_mode}")

                    if return_pil:
                        images.append(image)
                    else:
                        buffered = BytesIO()
                        image.save(buffered, format="PNG")
                        img_b64_str = base64.b64encode(buffered.getvalue()).decode()
                        images.append(img_b64_str)
        return images

    def to_gradio_chatbot(self):
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    import base64
                    from io import BytesIO
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

    def dict(self):
        if len(self.get_images()) > 0:
            return {
                "system": self.system,
                "roles": self.roles,
                "messages": [[x, y[0] if type(y) is tuple else y] for x, y in self.messages],
                "offset": self.offset,
                "sep": self.sep,
                "sep2": self.sep2,
            }
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
        }


conv_bunny = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the user's questions.",
    roles=("USER", "ASSISTANT"),
    version="bunny",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="<|endoftext|>",
)

conv_phi3 = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the user's questions.",
    roles=("USER", "ASSISTANT"),
    version="phi3",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="<|endoftext|>",
)

conv_minicpm = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the user's questions.",
    roles=("USER", "ASSISTANT"),
    version="minicpm",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)

conv_llama = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the user's questions.",
    roles=("USER", "ASSISTANT"),
    version="llama",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="<|end_of_text|>",
)

conv_plain = Conversation(
    system="",
    roles=("", ""),
    messages=(
    ),
    offset=0,
    sep_style=SeparatorStyle.PLAIN,
    sep="\n",
)

default_conversation = conv_bunny
conv_templates = {
    "default": conv_bunny,
    "bunny": conv_bunny,
    "phi3": conv_phi3,
    "plain": conv_plain,
    'minicpm': conv_minicpm,
    'llama': conv_llama
}

if __name__ == "__main__":
    print(default_conversation.get_prompt())




# 这个 Conversation 类是一个用于多轮对话管理的容器，内部管理对话历史、角色、文本、图片等信息，并提供多种方法辅助处理和格式化对话内容。下面对你给出的几个方法做详细解释：

# 1. get_prompt(self)
# 作用：将对话的历史消息按特定格式拼接成字符串，用作模型的输入Prompt。

# 细节：

# 处理消息列表中可能存在复杂的图像数据，用特殊格式进行清理和调整。

# 根据 sep_style 选择提示符拼接风格：

# SeparatorStyle.TWO：采用两个分隔符交替拼接，并且在消息中显示角色（如 "human: 内容###"）。

# SeparatorStyle.PLAIN：只拼接消息内容和一个分隔符，不带角色。

# 支持对“多模态标签”的版本做特殊处理，插入图片标志和接收确认。

# 结果是连贯且格式化的字符串，方便送入语言模型。

# 2. append_message(self, role, message)
# 作用：向对话中追加一条新消息。

# 参数：

# role：消息说话者角色名（例如 "human" 或 "gpt"）。

# message：消息内容，字符串或特殊格式（比如包含图片信息的tuple）。

# 实现简单，就是添加一个 [role, message] 列表到 messages 末尾。

# 3. get_images(self, return_pil=False)
# 作用：从对话历史中提取所有的图片消息。

# 实现重点：

# 遍历从 offset 开始的消息，过滤出索引为偶数（人类消息或者特殊含图消息）。

# 判断消息内容是否是tuple类型（说明里面带图片数据）。

# 根据image_process_mode，对PIL图片进行处理（如裁剪、填充、调整尺寸）。

# 返回结果可根据参数 return_pil 决定是直接返回PIL对象，还是转成Base64字符串。

# 方便前端渲染，或后续图像处理。

# 4. to_gradio_chatbot(self)
# 作用：将对话转换成适合Gradio聊天控件显示的格式。

# 细节：

# 遍历消息，从offset开始。

# 偶数消息（人类侧）如果是带图片的元组，做图片压缩、图像比例缩放，转Base64编码。

# 形成HTML格式的图片元素，和纯文本消息统一封装成列表 [消息, None] 。

# 奇数消息（机器回答）则补充前一条消息中的第二元素（通常是回答文本）。

# 最终形成列表，方便Gradio页面逐步渲染对话。

# 5. copy(self)
# 作用：拷贝当前对话对象内容，生成一个全新的 Conversation 实例副本。

# 防止外部修改影响原始对象，用于保存快照或并行操作。

# 包括系统提示信息、角色、消息列表（深拷贝）、偏移量、分隔符配置等成员变量。

# 6. dict(self)
# 作用：把对话对象序列化成字典格式，方便存储或传输。

# 细节：

# 如果对话中存在图片，消息中的每条内容如果是tuple时只保留文本部分（第一项）。

# 不含图片时直接返回全部消息。

# 返回的字典可用于JSON编码、日志记录、API交互等场景。

# 总结
# 这个类的核心职责是：

# 管理对话历史，包括文本和图像消息；

# 根据业务需求和输出格式，对对话历史做格式化处理（字符串拼接、图片提取、界面格式转换等）；

# 为多模态对话模型提供合适的上下文输入构建；

# 方便对话的扩展、复制和序列化。

# 所有方法都是围绕以上目标设计，紧密协作，支持复杂多轮、多模态对话的实现。



# 这两段代码的作用其实有本质区别，不是完全相同，关键区别在于拼接结果中是否添加了角色名称：

# 当 self.sep_style == SeparatorStyle.TWO 时：

# 拼接的对话字符串中每个消息前都会拼上角色名，比如：

# text
# human: <message>###
# gpt: <message>+++
# human: <message>###
# ...
# 其中，seps = [self.sep, self.sep2] 是两个分隔符交替使用，保证对话轮次清晰分隔。

# 拼接时对于消息为空的，至少会拼接 "role:" 用于占位。

# 当 self.sep_style == SeparatorStyle.PLAIN 时：

# 拼接时不拼接角色名，只直接用消息文本和分隔符拼接，比如：

# text
# <message>###
# <message>+++
# ...
# 对消息为空的情况，不拼接任何内容，完全忽略。

# 为什么设计成这样？
# SeparatorStyle.TWO 适用于需要明确区分说话角色的多轮对话场景，显示更明确的“谁说的什么”，方便模型理解。

# SeparatorStyle.PLAIN 适用于某些场景对角色不敏感，只拼接连续文本内容，简化输入格式