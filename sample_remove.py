import os
import random
import shutil

def move_random_20_percent(src_folder, dst_folder, seed=42):
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.raw')
    os.makedirs(dst_folder, exist_ok=True)

    image_files = [
        f for f in os.listdir(src_folder)
        if f.lower().endswith(valid_exts)
    ]

    total = len(image_files)
    if total == 0:
        print(f"⚠️ {src_folder} 没有找到图片，跳过。")
        return

    # 计算要移动的数量（比例 total * 0.2 ）
    total = len(image_files)
    num_to_move = max(1, int(total * 0.17))
    print(f"Found {total} images, moving {num_to_move} randomly...")

    # 固定随机种子，保证可复现
    random.seed(seed)
    selected_files = random.sample(image_files, num_to_move)

    # 执行移动操作
    for filename in selected_files:
        src_path = os.path.join(src_folder, filename)
        dst_path = os.path.join(dst_folder, filename)
        shutil.move(src_path, dst_path)
        print(f"Moved: {filename}")

    print("✅ Done.")


# 将源文件夹中 20% 的图片移动到目标文件夹中
if __name__ == "__main__":
    # src = "../train/MutiClassify/MultiDistanceAmp"
    # dst = "../test/MutiClassify/MultiDistanceAmp"
    src = "/Users/risetto/Documents/MATLAB/1550/小鼠卵泡数据集/疾病/001_opt"
    dst = "/Users/risetto/Documents/MATLAB/1550/小鼠卵泡数据集/疾病/001_opt/valid"
    move_random_20_percent(src, dst)

