import os
from shutil import copy2
import tqdm
from joblib import Parallel, delayed

# 所有图片所在位置（自己改）
image_dir = "../../images/"
data = os.listdir(image_dir)
get_key = lambda i : i.split('.')[0]
all_data = sorted(data, key=get_key)
num_all_data = len(all_data)
# for i in range(num_all_data):
#     print(all_data[i])


trainDir = "../../data/images/train"  # （将训练集放在这个文件夹下）
if not os.path.exists(trainDir):
    os.mkdir(trainDir)
validDir = '../../data/images/val'  # （将验证集放在这个文件夹下）
if not os.path.exists(validDir):
    os.mkdir(validDir)
testDir = '../../data/images/test'  # （将测试集放在这个文件夹下）
if not os.path.exists(testDir):
    os.mkdir(testDir)


def split_images(i):
    if i < num_all_data * 0.75:
        copy2(image_dir + all_data[i], trainDir)
    elif i < num_all_data * 0.95:
        copy2(image_dir + all_data[i], validDir)
    else:
        copy2(image_dir + all_data[i], testDir)


if __name__ == '__main__':
    Parallel(n_jobs=8)(delayed(split_images)(i) for i in tqdm.tqdm(range(0, num_all_data)))
