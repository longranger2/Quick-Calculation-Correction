import torch
import torch.nn as nn
import argparse
import crnn

from crnn import CRNN
from utils.aftertreatment import StrLabelConverter
from utils.datasets import get_DataLoader
from utils.fileoperation import get_chinese
from utils.loggers.log import log_test, log_device, log_load_model


def test(crnn, test_iter, criterion, device, converter, flag=True):
    total_num = 0.
    total_loss = 0.
    total_acc = 0.
    for i in range(len(test_iter)):
        crnn.eval()
        acc_num = 0.
        images, labels = test_iter.next()
        y = labels
        images = images.to(device)
        batch_size = images.size(0)
        total_num += batch_size
        text, length = converter.encode(labels)

        with torch.no_grad():
            preds = crnn(images)
            preds_size = torch.IntTensor([preds.size(0)] * batch_size)

            loss = criterion(preds, text, preds_size, length)
            total_loss += loss.item()

            y_hat = nn.functional.softmax(preds, 2).argmax(2).view(preds.size(0), -1)
            y_hat = torch.transpose(y_hat, 1, 0)
            y_hat = [converter.decode(i, torch.IntTensor([y_hat.size(1)])) for i in y_hat]

            for txt, target in zip(y, y_hat):
                if txt != target:
                    print(txt, '------------>', target, end='   ')
                    print('\033[1;31m❌❌❌\033[0m')
                elif flag:
                    acc_num += 1
                    print(txt, '------------>', target, end='   ')
                    print('\033[1;32m✔✔✔\033[0m')
                else:
                    acc_num += 1

            total_acc += acc_num
        log_test(loss.item() / batch_size, acc_num / batch_size)
    log_test(total_loss / total_num, total_acc / total_num, False)

def main(opt):
    chinese = get_chinese(opt.chinese)
    converter = StrLabelConverter(chinese)
    nclass = len(chinese) + 1

    test_loader = get_DataLoader('test', opt)
    test_iter = iter(test_loader)
    criterion = nn.CTCLoss(reduction='sum')
    crnn = CRNN(opt.imgH, opt.nc, nclass, opt.nh)
    crnn.load_state_dict(torch.load(opt.weights))
    log_load_model(opt.weights)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_device(device)

    crnn = crnn.to(device)
    criterion = criterion.to(device)

    test(crnn, test_iter, criterion, device, converter, opt.all)


def parse_opt():
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--weights', type=str, default='weights/CPU.pt', help='权重的路径')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--all', action='store_true', default=True)
    parser.add_argument('--chinese', type=str, default='data/formula.txt', help='字符集保存路径')
    parser.add_argument('--images', type=str, default='data/images/', help='图片路径，同train.py的参数一样')
    parser.add_argument('--labels', type=str, default='data/labels/', help='标签路径')
    parser.add_argument('--imgH', type=int, default=32)
    parser.add_argument('--nc', type=int, default=1)
    parser.add_argument('--nh', type=int, default=256)
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
