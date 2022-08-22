import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from cifar_resnet import ResNet18
import cifar_resnet as resnet

import torch_pruning as pruning
import argparse
import torch
from torchvision.datasets import CIFAR10
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn 
import numpy as np 

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, required=True, choices=['train', 'prune', 'test'])
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--verbose', action='store_true', default=False)
parser.add_argument('--total_epochs', type=int, default=100)
parser.add_argument('--step_size', type=int, default=70)
parser.add_argument('--round', type=int, default=1)

args = parser.parse_args()

def get_dataloader():
    train_loader = torch.utils.data.DataLoader(
        CIFAR10('./data', train=True, transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]), download=True),batch_size=args.batch_size, num_workers=2)
    test_loader = torch.utils.data.DataLoader(
        CIFAR10('./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
        ]),download=True),batch_size=args.batch_size, num_workers=2)
    return train_loader, test_loader

def eval(model, test_loader):
    correct = 0
    total = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    with torch.no_grad():
        for i, (img, target) in enumerate(test_loader):
            img = img.to(device)
            out = model(img)
            pred = out.max(1)[1].detach().cpu().numpy()
            target = target.cpu().numpy()
            correct += (pred==target).sum()
            total += len(target)
    return correct / total

def train_model(model, train_loader, test_loader):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.step_size, 0.1)
    model.to(device)

    best_acc = -1
    for epoch in range(args.total_epochs):
        model.train()
        for i, (img, target) in enumerate(train_loader):
            img, target = img.to(device), target.to(device)
            optimizer.zero_grad()
            out = model(img)
            loss = F.cross_entropy(out, target)
            loss.backward()
            optimizer.step()
            if i%10==0 and args.verbose:
                print("Epoch %d/%d, iter %d/%d, loss=%.4f"%(epoch, args.total_epochs, i, len(train_loader), loss.item()))
        model.eval()
        acc = eval(model, test_loader)
        print("Epoch %d/%d, Acc=%.4f"%(epoch, args.total_epochs, acc))
        if best_acc<acc:
            torch.save( model, 'resnet18-round%d.pth'%(args.round) )
            best_acc=acc
        scheduler.step()
    print("Best Acc=%.4f"%(best_acc))

def prune_model_with_shortcut(model):
    model.cpu()
    print(model)
    slim=pruning.Autoslim(model,inputs=torch.randn(1,3,32,32),compression_ratio=0.795)
    slim.l1_norm_pruning()
    print('---------------------')
    print(model)

    return model    

def prune_model_without_shortcut(model):
    model.cpu()
    slim=pruning.Autoslim(model,inputs=torch.randn(1,3,32,32),compression_ratio=0)
    
    
    #print(model)
    layer_compression_rate={5:0.2,11:0.5,18:0.5,26:0.5,33:0.75,41:0.75,48:0.875,56:0.875}
    slim.l1_norm_pruning(layer_compression_ratio=layer_compression_rate)
    #print(model)
    return model

def prune_model_mixed(model):

    model.cpu()
    slim=pruning.Autoslim(model,inputs=torch.randn(1,3,32,32),compression_ratio=0)
    layer_compression_rate={}
    layer_index_record=[]
    for key,value in slim.index_of_layer().items():
        layer_index_record.append(key)
        #print(key,value)
    for num in layer_index_record:
        if num<18:
            layer_compression_rate[num]=0.5
        elif num<33:
            layer_compression_rate[num]=0.625
        elif num<48:
            layer_compression_rate[num]=0.75
        else:
            layer_compression_rate[num]=0.875

    
    #slim.l1_norm_pruning(layer_compression_ratio=layer_compression_rate)
    slim.l1_norm_pruning()
    print('---------------------------')
    return model


def main():
    train_loader, test_loader = get_dataloader()
    if args.mode=='train':
        args.round=0
        model = ResNet18(num_classes=10)
        train_model(model, train_loader, test_loader)
    elif args.mode=='prune':
        previous_ckpt = 'resnet18-round%d.pth'%(args.round-1)
        print("Pruning round %d, load model from %s"%( args.round, previous_ckpt ))
        model = torch.load( previous_ckpt )

        prune_model_with_shortcut(model)
        #prune_model_without_shortcut(model)
        #prune_model_mixed(model)
        #print(model)
        params = sum([np.prod(p.size()) for p in model.parameters()])
        print("Number of Parameters: %.3fM"%(params/1e6))
        train_model(model, train_loader, test_loader)

    elif args.mode=='test':
        ckpt = 'resnet18-round%d.pth'%(args.round)
        print("Load model from %s"%( ckpt ))
        model = torch.load( previous_ckpt )
        params = sum([np.prod(p.size()) for p in model.parameters()])
        print("Number of Parameters: %.1fM"%(params/1e6))
        acc = eval(model, test_loader)
        print("Acc=%.4f\n"%(acc))

if __name__=='__main__':
    main()
