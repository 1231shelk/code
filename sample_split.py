import os
import shutil

# 从源文件夹中筛选所有文件名以 Healthy 开头的图像文件，并复制到（保留源文件）目标文件夹

# 源文件夹\目标文件夹
source_folder = '../train/001'
target_folder = '../Healthy001'

# 支持的图片扩展名
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')

# 遍历文件夹中的所有文件
for filename in os.listdir(source_folder):
    if filename.startswith('Healthy') and filename.endswith(image_extensions):
        source_path = os.path.join(source_folder, filename)
        target_path = os.path.join(target_folder, filename)
        shutil.copy2(source_path, target_path)
        print(f"已移动: {filename}")

print("处理完成！")

