'''
# Adapted from https://github.com/baaivision/EVA/tree/master/EVA-CLIP
'''

from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from transformers.image_processing_utils import BatchFeature
from PIL import Image
from transformers.image_transforms import convert_to_rgb


#
#初始化了一个transform属性，它是一个函数，默认是恒等函数，即输入什么返回什么。

class BaseProcessor:
    def __init__(self):
        self.transform = lambda x: x
        return

    def __call__(self, item):
        return self.transform(item)

#图像的每个通道（一般是RGB三个通道）的像素值都需要分别进行归一化处理。
class EvaClipImageBaseProcessor(BaseProcessor):
    def __init__(self, mean=None, std=None):
        self.mean = (0.48145466, 0.4578275, 0.40821073) if mean is None else mean
        self.std = (0.26862954, 0.26130258, 0.27577711) if std is None else std

        self.normalize = transforms.Normalize(self.mean, self.std)

    @property
    def image_mean(self):
        return self.mean


class EvaClipImageTrainProcessor(EvaClipImageBaseProcessor):
    def __init__(self, image_size=224, mean=None, std=None, min_scale=0.5, max_scale=1.0):
        super().__init__(mean=mean, std=std)

        self.transform = transforms.Compose(
            [
                convert_to_rgb,
                transforms.Resize(
                    image_size,
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                self.normalize,
            ]
        )

        self.image_size = image_size

    def preprocess(self, images, return_tensors):
        if isinstance(images, Image.Image):
            images = [images]
        else:
            assert isinstance(images, list)

        transformed_images = [self.transform(image).numpy() for image in images]
        data = {"pixel_values": transformed_images}

        return BatchFeature(data=data, tensor_type=return_tensors)

    def __call__(self, item):
        return self.transform(item)

    @property
    def crop_size(self):
        return {'height': self.image_size, 'width': self.image_size}



# 这个类 EvaClipImageTrainProcessor 是CLIP模型图像输入数据的训练时预处理器，主要作用是将输入的PIL图像转换成适合CLIP视觉编码器的格式。它的设计和处理流程如下：

# 主要功能和处理流程
# 继承自基础归一化处理类
# 继承了EvaClipImageBaseProcessor，利用其定义好的图像归一化（normalize）功能，保证输入图像像素值标准化。

# 定义图像预处理管线（transform）
# 使用torchvision.transforms.Compose串接多个预处理步骤：

# convert_to_rgb：确保图像是RGB格式，避免灰度图或其他格式带来的输入异常。

# Resize(image_size, interpolation=BICUBIC)：把图像调整为指定的输入尺寸（默认为224），使用双三次插值保证图像细节。

# CenterCrop(image_size)：从调整后图像中间裁剪出指定大小，保证输入尺寸固定且聚焦图像中心区域。

# ToTensor()：把PIL图像转换成PyTorch张量，数据形状从(H, W, C)变为(C, H, W)，且像素值从0-255归一到0-1。

# self.normalize：采用在基类定义的均值和标准差对三个通道独立归一化，得到模型训练时所需的标准像素分布。

# 支持批量预处理
# preprocess函数支持输入单张或多张PIL图像，统一转换为模型可接受的Tensor格式，并返回封装后的BatchFeature（便于模型输入）。

# 重写类的调用接口
# __call__方法返回单张图片的预处理结果，实现调用实例像函数一样直接用在图像上。

# 提供裁剪大小信息
# crop_size属性以字典形式返回图像最终的裁剪尺寸，供外部参考使用。

# 设计意义和合理性
# 保证输入统一：调整尺寸和中心裁剪保证所有输入图像尺寸一致，这对Transformer类视觉模型非常关键，因为Transformer需要固定长度序列作为输入。

# 保留关键细节：用双三次插值放缩图像，中心裁剪避免边界干扰，同时保留图像中部更重要的内容。

# 标准化输入分布：归一化像素值使得图像数据分布与模型训练阶段保持一致，提高模型泛化能力和鲁棒性。

# 接口友好：支持单张和多张图片处理，多图支持方便批量训练，返回BatchFeature统一格式方便集成。

# 灵活配置：支持动态输入尺寸设置和均值、标准差定制，方便适配不同模型需求。

# 综上，这是一个典型CLIP视觉模型训练阶段图像预处理设计，合理串联了尺寸调整、裁剪、格式转换到标准化的流程，确保输入图像符合模型对尺寸和数据分布的一致性要求。这个类 EvaClipImageTrainProcessor 是针对CLIP类视觉模型训练时图像预处理的模块，主要完成以下功能和流程：

# 继承了基础归一化类 EvaClipImageBaseProcessor，继承了图像标准化的均值和标准差，并定义了归一化变换 self.normalize。

# 它的transform是由多个步骤组合构成：

# convert_to_rgb：确保输入图像为RGB格式。

# Resize(image_size, interpolation=InterpolationMode.BICUBIC)：将图像缩放至指定大小（如224×224），采用双三次插值保持图像细节。

# CenterCrop(image_size)：从缩放后的图像中心裁剪出指定大小的区域，确保输入尺寸固定。

# ToTensor()：将图像转为PyTorch张量，且像素范围从0-255转为0-1。

# self.normalize：用CLIP训练时的均值和标准差对每个通道像素做归一化。

# 定义了preprocess函数，支持单张或多张PIL图像输入，做相同transform后转换成模型接受的Tensor格式，并封装成BatchFeature方便输入。

# __call__重载使得对象可直接对单张图像调用完成预处理。

# crop_size属性告诉外部裁剪后图像的宽高信息。

# 设计意图是为了确保训练阶段输入的图像：

# 尺寸统一且符合模型要求（防止输入大小不一致导致模型崩溃或性能下降），

# 保持中心信息，减少边界噪声影响，

# 像素范畴和分布匹配预训练模型输入，提高训练收敛和泛化能力。

# 这种设计是经典的视觉模型图像处理流程，尤其是Vision Transformer类模型对输入尺寸和格式要求较严，它保证了训练数据质量和一致性，方便模型训练和微调。