import os
from PIL import Image

def resize_and_rename_images(src_folder, dst_folder, size=(224, 224)):
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')

    os.makedirs(dst_folder, exist_ok=True)

    # 统计目标文件夹中已有图片数量
    existing_images = [
        f for f in os.listdir(dst_folder)
        if f.lower().endswith(valid_exts)
    ]
    start_index = len(existing_images) + 1

    image_files = sorted([
        f for f in os.listdir(src_folder)
        if f.lower().endswith(valid_exts)
    ])

    for idx, filename in enumerate(image_files, start=start_index):
        name, ext = os.path.splitext(filename)
        ext = ext.lower()
        # new_name = "Healthy" + f"{idx:05d}{ext}"
        # new_name = "Disease" + f"{idx:05d}{ext}"
        # new_name = "NCD137PCOS" + f"{idx:05d}{ext}"  # 001
        # new_name = "NCD139PCOSICR" + f"{idx:05d}{ext}"  # 002
        # new_name = "NCD72SQTLC57" + f"{idx:05d}{ext}"  # 003
        # new_name = "MODEL79SQTLC57" + f"{idx:05d}{ext}"  # 001
        new_name = "MODEL2SQTLC5" + f"{idx:05d}{ext}"  # 003
        src_path = os.path.join(src_folder, filename)
        dst_path = os.path.join(dst_folder, new_name)

        try:
            with Image.open(src_path) as img:
                img_resized = img.resize(size, Image.Resampling.LANCZOS)
                img_resized.save(dst_path)
            print(f"Saved resized image: {new_name}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")


# 批量读取一个文件夹中的图片,统一调整尺寸(默认224×224),并以统一的格式重命名保存到目标文件夹中
if __name__ == "__main__":
    src = "/Users/risetto/Documents/MATLAB/1550/小鼠卵泡数据集/疾病组/003_opt_res/MultiDistanceAmp/train"
    dst = "../data/MultiDistanceAmpTrain"
    resize_and_rename_images(src, dst)

