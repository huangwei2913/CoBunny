# 阶段2：微调（包含 Recipe-1 和 Recipe-2）

#可以把 Recipe-1 和 Recipe-2 写在一个脚本里，用 && 连接，确保第一步成功后自动跑第二步：

#Recipe-1：--unfreeze_vision_tower False。先让语言模型学会多模态指令。

#Recipe-2：--unfreeze_vision_tower True。打开视觉塔，微调全链路。

#关键修正：全部统一使用 --version bunny，彻底告别 phi3。