import torch
import torch.nn as nn
import tqdm
from utils.loggers.log import log_epoch


def val(crnn, val_iter, criterion, device, converter, epoch):
    total_num = 0.
    total_loss = 0.
    total_acc = 0.
    with tqdm.tqdm(range(len(val_iter))) as tbar:
        for i in tbar:
            crnn.eval()
            acc_num = 0.
            images, labels = val_iter.next()
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
                    if txt == target:
                        acc_num += 1
                total_acc += acc_num

            tbar.set_description('epoch {}'.format(epoch))
            tbar.set_postfix(loss=loss.item() / batch_size, acc=acc_num / batch_size)
            tbar.update()
    log_epoch(epoch, total_loss / total_num, total_acc / total_num, 'val')
    return total_loss / total_num, total_acc / total_num
