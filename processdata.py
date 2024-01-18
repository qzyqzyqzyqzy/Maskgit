import os
import shutil
from PIL import Image
import torch
from torchvision import transforms
import torchvision
# 定义你的数据集路径
imagenet_dir = '../tiny200/tiny-imagenet-200/train'
output_dir = './owndataset1/class1'

# 创建输出目录
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 定义一个函数来处理和保存图像
def process_and_save_images(dataset_dir):
    for root, dirs ,_ in os.walk(dataset_dir):
        for subdir in dirs:
            subdir_path = os.path.join(root, subdir)
            for root_sub,_, files_sub in os.walk(subdir_path):
                for file_sub in files_sub:
                    if file_sub.endswith('.jpeg') or file_sub.endswith('.JPEG'):  # 根据你的数据集中的图像格式修改
                        image_path = os.path.join(root_sub, file_sub)
                        try:
                            img = Image.open(image_path)
                            img.verify()  # 验证图像，如果无法打开，则会抛出异常
                        # 将图像复制到输出目录
                            shutil.copy(image_path, output_dir)
                        except (IOError, SyntaxError) as e:
                            print('Bad file:', image_path)  # 输出损坏文件的路径

# 处理并保存ImageNet和CIFAR100的图像
process_and_save_images(imagenet_dir)

cifar100=torchvision.datasets.CIFAR100(root='./cifadata',train=True,download=True)
save_folder = './owndataset2/class2'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# 遍历CIFAR100的所有图片并保存到文件夹
for i, (image, label) in enumerate(cifar100):
    image_path = os.path.join(save_folder, f'image_{i}_{label}.jpg')
    image.save(image_path)