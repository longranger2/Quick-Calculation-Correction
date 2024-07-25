from PIL import Image
import os
from shutil import copy2
import tqdm
from joblib import Parallel, delayed

# 所有图片所在位置（自己改）
image_dir = "../../images/"
data = os.listdir(image_dir)
num=len(data)
sum_w=0
sum_h=0

for i in range(num):
    im=Image.open("../../images/"+data[i])
    w,h=im.size
    sum_w+=w
    sum_h+=h
print(sum_w/num)#所有图片的平均宽度
print(sum_h/num)#所有图片的平均高度

# im = Image.open(filename)#返回一个Image对象
# print('宽：%d,高：%d'%(im.size[0],im.size[1]))