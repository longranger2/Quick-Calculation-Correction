import numpy as np
from PIL import Image
from torchvision import transforms


class ResizeAndNormalize(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):

        size = self.size
        imgW, imgH = size
        # 等比例放大或缩小图片
        scale = img.size[1] * 1.0 / imgH
        w = img.size[0] / scale
        w = int(w)
        img = img.resize((w, imgH), self.interpolation)
        w, h = img.size
        # 图片宽度小于需要的宽度，则在右方补充255，否则，直接resize
        if w <= imgW:
            newImage = np.zeros((imgH, imgW), dtype='uint8')
            newImage[:] = 255
            newImage[:, :w] = np.array(img)
            img = newImage
        else:
            img = img.resize((imgW, imgH), self.interpolation)

        # 转换成Tensor
        img = self.toTensor(img)
        # 让图像分布在(-1，1)之间
        img.sub_(0.5).div_(0.5)
        return img
