import torch
import torch.nn as nn


def train_batch(crnn, train_iter, optimizer, criterion, device, converter):
    acc_num = 0
    images, labels = train_iter.next()
    y = labels

    images = images.to(device)
    batch_size = images.size(0)
    text, length = converter.encode(labels)

    preds = crnn(images)
    preds_size = torch.IntTensor([preds.size(0)] * batch_size)

    loss = criterion(preds, text, preds_size, length)

    y_hat = nn.functional.softmax(preds, 2).argmax(2).view(preds.size(0), -1)
    y_hat = torch.transpose(y_hat, 1, 0)
    y_hat = [converter.decode(i, torch.IntTensor([y_hat.size(1)])) for i in y_hat]

    for txt, target in zip(y, y_hat):
        if txt == target:
            acc_num += 1

    crnn.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item(), batch_size, acc_num