import os
from PIL import Image, ImageDraw, ImageFont

def get_six_sub_images(image_path, target_sz=384):
    """
    高清六子图切分测试函数
    逻辑：1个全局缩放图 + 5个动态锚点高清切片
    """
    # 1. 加载图像
    raw_image = Image.open(image_path).convert('RGB')
    w, h = raw_image.size
    print(f"原始图像尺寸: {w}x{h}")

    # --- 核心逻辑：锚点计算器 ---
    def calculate_anchors(full_len, target_len):
        if full_len <= target_len:
            return [0, 0, 0, 0, 0]
        max_scroll = full_len - target_len
        return [
            0,                      # 起点 (左/上)
            max_scroll // 4,        # 1/4处
            max_scroll // 2,        # 中点
            3 * max_scroll // 4,    # 3/4处
            max_scroll              # 终点 (右/下)
        ]

    # 计算横向和纵向的采样起点
    x_coords = calculate_anchors(w, target_sz)
    y_coords = calculate_anchors(h, target_sz)

    # 2. 生成全局图 (Global View)
    global_img = raw_image.resize((target_sz, target_sz), Image.BILINEAR)

    # 3. 产生 5 个局部高清切片 (Local Views)
    # 我们按照：左上、右上、左下、右下、正中心的顺序排列
    # 注意：坐标索引 0 是起点，4 是终点，2 是中点
    crops = [
        raw_image.crop((x_coords[0], y_coords[0], x_coords[0]+target_sz, y_coords[0]+target_sz)), # 左上
        raw_image.crop((x_coords[4], y_coords[0], x_coords[4]+target_sz, y_coords[0]+target_sz)), # 右上
        raw_image.crop((x_coords[0], y_coords[4], x_coords[0]+target_sz, y_coords[4]+target_sz)), # 左下
        raw_image.crop((x_coords[4], y_coords[4], x_coords[4]+target_sz, y_coords[4]+target_sz)), # 右下
        raw_image.crop((x_coords[2], y_coords[2], x_coords[2]+target_sz, y_coords[2]+target_sz)), # 绝对中心
    ]

    return [global_img] + crops

def visualize_results(image_path, output_path="test_result.jpg"):
    """
    可视化函数：将 6 个子图拼成 2x3 的大图进行查看
    """
    target_sz = 384
    sub_images = get_six_sub_images(image_path, target_sz)
    
    # 创建一张大画布 (2行3列)
    combined = Image.new('RGB', (target_sz * 3, target_sz * 2))
    labels = ["Global", "Top-Left", "Top-Right", "Bottom-Left", "Bottom-Right", "Center"]
    
    draw = ImageDraw.Draw(combined)
    
    for i, img in enumerate(sub_images):
        x = (i % 3) * target_sz
        y = (i // 3) * target_sz
        combined.paste(img, (x, y))
        # 在子图左上角画个简单的标识文字（可选）
        draw.text((x + 10, y + 10), labels[i], fill=(255, 0, 0))

    combined.save(output_path)
    print(f"可视化测试结果已保存至: {output_path}")

# --- 测试运行 ---
if __name__ == "__main__":
    # 请替换为你本地的一张高清测试图路径（尤其是那种长条图或者大尺寸手机拍照图）
    test_image = "/mnt/CoBunny/bunny/model/language_model/gupiao.jpg" 
    
    if os.path.exists(test_image):
        visualize_results(test_image)
    else:
        print(f"请先准备一张名为 {test_image} 的图片进行测试！")