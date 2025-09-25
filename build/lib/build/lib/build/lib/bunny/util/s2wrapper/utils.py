#  ------------------------------------------------------------------------------------------
#  Copyright (c) 2024 Baifeng Shi.
#  All rights reserved.
#
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import torch


#这个表示分成多少个小块，假定 输入是224*224的时候，由于 num_split=16，图像被划分成 16×16=256 个小块，每个块大小是14*14
#split_chessboard 就是把大图像按网格切成多个小分片，然后把这些小分片沿批次维度拼接起来，从而增加批次数量，方便对这些小分片分别进行处理。
def split_chessboard(x, num_split):
    """
        x: b * c * h * w
        Deividing x into num_split**2 sub-squares, and concatenate all the sub-squares on the batch dimension
    """
    B, C, H, W = x.shape
    assert H % num_split == 0 and W % num_split == 0
    h, w = H // num_split, W // num_split
    x_split = torch.cat([x[:, :, i*h:(i+1)*h, j*w:(j+1)*w] for i in range(num_split) for j in range(num_split)], dim=0)
    return x_split



#split_chessboard的逆向操作，还原成原来的输入 b * c * h * w
def merge_chessboard(x, num_split):
    """
        x: b * c * h * w
        Assuming x contains num_split**2 sub-squares concatenated along batch dimension, merge the sub-squares back to the original whole square.
        (inverse of split_chessboard)
    """
    B, C, H, W = x.shape
    assert B % (num_split**2) == 0
    b = B // (num_split**2)
    x_merge = torch.cat([torch.cat([x[(i*num_split + j)*b:(i*num_split + j + 1)*b] for j in range(num_split)], dim=-1)
                         for i in range(num_split)], dim=-2)
    return x_merge



# x.split(batch_size)：这是 PyTorch 的张量切分操作，
# 沿第一个维度（batch维）将输入张量分成多个小张量，每个大小为 batch_size，返回一个张量列表。例如总共有100个样本，batch_size=20，则拆成5个张量，每个包含20个样本。
def batched_forward(model, x, batch_size=-1):
    if batch_size == -1:
        return model(x)
    else:
        x_batched = x.split(batch_size)
        outs = [model(x) for x in x_batched]
        return torch.cat(outs, dim=0)




