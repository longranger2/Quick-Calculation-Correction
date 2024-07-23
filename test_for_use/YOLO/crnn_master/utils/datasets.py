from PIL import Image
from torch.utils.data import Dataset, DataLoader
import os

from utils.pretreatment import ResizeAndNormalize


class CRNNDataset(Dataset):

    def __init__(self, image_path, label_path):
        self.image_path = image_path
        self.label_path = label_path
        self.image_dict = self.readfile()
        self.image_name = [filename for filename, _ in self.image_dict.items()]

    def __getitem__(self, index):
        # 图片路径
        image_path = os.path.join(self.image_path, self.image_name[index])
        # 图片标签
        label = self.image_dict.get(self.image_name[index])
        # 读取灰度图的图片
        image = Image.open(image_path).convert('L')
        # 对图片进行resize和归一化操作
        transform = ResizeAndNormalize((140, 32))
        image = transform(image)
        return image, label

    def __len__(self):
        return len(self.image_dict)

    def readfile(self):
        dir = {}
        with open(self.label_path, encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                k = line.strip().replace('\n', '').split(r' ')[0]
                v = line.strip().replace('\n', '').split(r' ')[1]
                dir[k] = v
            f.close()
        return dir


def get_DataLoader(mode, opt):
    if os.path.exists(opt.images + mode):
        dataset = CRNNDataset(opt.images + mode + '/', opt.labels + mode + '.txt')
    else:
        dataset = CRNNDataset(opt.images + '/', opt.labels + mode + '.txt')
    flag = True
    if mode != 'train':
        flag = False
    loader = DataLoader(dataset=dataset, batch_size=opt.batch_size, shuffle=flag)
    return loader





