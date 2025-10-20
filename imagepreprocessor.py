# Copyright 2024 Your Organization
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def lcm(a, b):
    return abs(a*b) // gcd(a, b)

def list_lcm(numbers):
    from functools import reduce
    return reduce(lcm, numbers)

class ImageProcessor:
    def __init__(self, patch_size_list, max_size=1152, min_no_scale=384):
        self.patch_size_list = patch_size_list
        self.max_size = max_size
        self.min_no_scale = min_no_scale
        self.patch_lcm = list_lcm(patch_size_list)

    def process_image(self, image: torch.Tensor) -> torch.Tensor:
        # image shape: (C, H, W)
        C, H, W = image.shape

        # 如果图像小于最小不处理尺寸直接返回
        if H <= self.min_no_scale and W <= self.min_no_scale:
            return image

        # 384 到 1152 之间，不处理
        if self.min_no_scale < H <= self.max_size and self.min_no_scale < W <= self.max_size:
            return image

        # 大于1152，插值到<=1152且为最小公倍数倍数的最大尺寸
        if H > self.max_size or W > self.max_size:
            scale_H = self.max_size // self.patch_lcm
            new_H = self.patch_lcm * scale_H
            scale_W = self.max_size // self.patch_lcm
            new_W = self.patch_lcm * scale_W
            image = F.interpolate(image.unsqueeze(0), size=(new_H, new_W), mode='bilinear', align_corners=True).squeeze(0)
            return image

        # 小于1152的且不在不处理范围（基本是小于384的情况)
        # 判断能否被最小公倍数整除，不能就缩放
        if H % self.patch_lcm != 0 or W % self.patch_lcm != 0:
            new_H = (H // self.patch_lcm) * self.patch_lcm
            new_W = (W // self.patch_lcm) * self.patch_lcm
            image = F.interpolate(image.unsqueeze(0), size=(new_H, new_W), mode='bilinear', align_corners=True).squeeze(0)

        return image


# 用法示例
#patch_size_list = [14, 16]
#processor = ImageProcessor(patch_size_list)
# processed_img = processor.process_image(input_image_tensor)
