# Model Constants
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
#DEFAULT_IMAGE_TOKEN = "<image>"  # 词汇表名称已经被污染了
DEFAULT_IMAGE_TOKEN = "<img_content>"  #这个预计的是50296  #专门负责承载图像特征。
IMAGE_SPLIT_TOKEN = "<pad>" #这个预计的是50297
CONTROLLER_HEART_BEAT_EXPIRATION = 30
LOGDIR = "gradio-logs"
WORKER_HEART_BEAT_INTERVAL = 15


