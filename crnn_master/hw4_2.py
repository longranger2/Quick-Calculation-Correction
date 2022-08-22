import os

import torch
import torch.nn as nn
import argparse
from crnn import CRNN
from crnn_master.utils.aftertreatment import StrLabelConverter
from crnn_master.utils.fileoperation import get_chinese
from crnn_master.utils.loggers.log import log_device, log_load_model
from PIL import Image

from crnn_master.utils.pretreatment import ResizeAndNormalize

def detect(model, image_path, device, converter):
    # 为灰度图像，每个像素用8个bit表示，0表示黑，255表示白，其他数字表示不同的灰度。
    image = Image.open(image_path).convert('L')
    # 对图片进行resize和归一化操作
    transform = ResizeAndNormalize((262, 32))
    image = transform(image)
    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        model.eval()
        preds = model(image)

        # print(preds.shape)
    preds = nn.functional.softmax(preds, 2).argmax(2).view(-1)
    # print(preds.shape)

    # 转成字符序列
    preds_size = torch.IntTensor([preds.size(0)])
    # print(preds_size)
    # print(preds)
    # raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    # print('%-20s => %-20s' % (raw_pred, sim_pred))
    #print(image_path, '---------->', sim_pred)
    return sim_pred

def detect_(crnn, source, device, converter):
    equations = []
    if os.path.isfile(source): #判断某一对象(需提供绝对路径)是否为文件
        source_ = source.lower() #将字母转为小写
        if source_.endswith('jpg') or source_.endswith('jpeg') or source_.endswith('png') or source_.endswith('bmp') or source_.endswith('tif') or source_.endswith('gif'):
            equations.append(detect(crnn, source, device, converter))

    else:#对象为文件夹
        source_list = os.listdir(source)
        source_list.sort() #升序排列
        for s in source_list:
            equations.append(detect(crnn, source + s, device, converter))
    return equations

def main(opt):
    chinese = get_chinese(opt.chinese)
    converter = StrLabelConverter(chinese)
    nclass = len(chinese) + 1

    crnn = CRNN(opt.imgH, opt.nc, nclass, opt.nh)
    crnn.load_state_dict(torch.load(opt.weights))
    log_load_model(opt.weights)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    log_device(device)
    crnn = crnn.to(device)
    equations = detect_(crnn, opt.source, device, converter)
    return equations

def parse_opt():
    parser = argparse.ArgumentParser(description='detect')
    parser.add_argument('--weights', type=str, default='../crnn_master/weights/CPU.pt', help='权重的路径')
    parser.add_argument('--source', type=str, default='../YOLO/yolo3/tmp_img/', help='要用来推理图片的路径，可以是一张图片，也可以是一个目录')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--chinese', type=str, default='../crnn_master/data/formula.txt', help='字符集保存路径')
    parser.add_argument('--imgH', type=int, default=32)
    parser.add_argument('--nc', type=int, default=1)
    parser.add_argument('--nh', type=int, default=256)
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
