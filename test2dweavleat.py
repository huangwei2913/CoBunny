import cv2
import numpy as np
import pywt
import matplotlib.pyplot as plt
import os

def save_coeff_image(coeff, title, filename):
    plt.imshow(coeff, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename)
    plt.clf()  # 清空当前图像，避免重叠

def dwt_image_decompose_save(image_path, wavelet='db1', level=2, save_dir='dwt_outputs'):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("图像读取失败，请检查路径")
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    coeffs = pywt.wavedec2(img, wavelet=wavelet, level=level)
    stats = []
    
    # 保存原图
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'original_image.png'))
    plt.clf()
    
    # 处理各层系数
    for i, coeff in enumerate(coeffs):
        if i == 0:
            # 最高层近似子带
            mean_val, var_val = np.mean(coeff), np.var(coeff)
            stats.append((f'Approximation Level {level}', mean_val, var_val))
            save_coeff_image(coeff, f'Approximation Level {level}\nmean={mean_val:.2f}, var={var_val:.2f}', 
                             os.path.join(save_dir, f'approximation_level_{level}.png'))
        else:
            # 细节子带，倒序对应层级，比如coeffs[1]是第level层的细节子带，依次往下
            cH, cV, cD = coeff
            curr_level = level - i + 1
            for name, band in zip(['Horizontal', 'Vertical', 'Diagonal'], [cH, cV, cD]):
                mean_val, var_val = np.mean(band), np.var(band)
                stats.append((f'Level {curr_level} {name} Detail', mean_val, var_val))
                save_coeff_image(band, f'Level {curr_level} {name} Detail\nmean={mean_val:.2f}, var={var_val:.2f}',
                                 os.path.join(save_dir, f'level_{curr_level}_{name.lower()}_detail.png'))
    
    # 打印均值和方差
    for name, mean_val, var_val in stats:
        print(f'{name}: mean={mean_val:.2f}, variance={var_val:.2f}')

# 使用示例
dwt_image_decompose_save('beignets-task-guide.png', wavelet='db1', level=2)
