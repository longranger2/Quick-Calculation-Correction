import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from cifar_resnet import ResNet18
import cifar100_resnet as resnet

import torch_pruning as pruning
import argparse
import torch
#from torchvision.datasets import CIFAR10
from torchvision.datasets import CIFAR100
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn 
import numpy as np 

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, required=True, choices=['train', 'prune', 'test'])
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--verbose', action='store_true', default=False)
parser.add_argument('--total_epochs', type=int, default=200)
parser.add_argument('--step_size', type=int, default=70)
parser.add_argument('--round', type=int, default=1)

args = parser.parse_args()

def get_dataloader():
    
    train_loader = torch.utils.data.DataLoader(
        CIFAR100('./data', train=True, transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]), download=True),batch_size=args.batch_size, num_workers=2)
    test_loader = torch.utils.data.DataLoader(
        CIFAR100('./data', train=False, transform=transforms.Compose([
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
    
    slim=pruning.Autoslim(model,inputs=torch.randn(1,3,32,32),compression_ratio=0.5)
    slim.l1_norm_pruning()
    # DG = tp.DependencyGraph().build_dependency( model, torch.randn(1, 3, 32, 32) )
    # def prune_conv(conv, pruned_prob):
    #     weight = conv.weight.detach().cpu().numpy()
    #     out_channels = weight.shape[0]
    #     L1_norm = np.sum( np.abs(weight), axis=(1,2,3))
    #     num_pruned = int(out_channels * pruned_prob)
    #     prune_index = np.argsort(L1_norm)[:num_pruned].tolist() # remove filters with small L1-Norm
    #     plan = DG.get_pruning_plan(conv, tp.prune_conv, prune_index)
    #     plan.exec()
    
    # block_prune_probs = [0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3]
    # blk_id = 0
    # for m in model.modules():
    #     if isinstance( m, resnet.BasicBlock ):
    #         prune_conv( m.conv1, block_prune_probs[blk_id] )
    #         prune_conv( m.conv2, block_prune_probs[blk_id] )
    #         blk_id+=1
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
    slim=pruning.Autoslim(model,inputs=torch.randn(1,3,32,32),compression_ratio=0.6)
    # layer_compression_rate={1:0.125,7:0.125,13:0.125,56:0.625}
    # layer_index_record=[]
    # for key,value in slim.index_of_layer().items():
    #     layer_index_record.append(key)
    #     print(key,value)

    # # 0.25 0.5 0.625 0.75 0.875
    # for num in layer_index_record:
    #     if num not in layer_compression_rate:
    #         if num<18:
    #             layer_compression_rate[num]=0.5
    #         elif num<33:
    #             layer_compression_rate[num]=0.625
    #         elif num<48:
    #             layer_compression_rate[num]=0.75
    #         else:
    #             layer_compression_rate[num]=0.875

    #??????????????????+????????????=????????????
    slim.fpgm_pruning()
    #print('layer_compression_rate:\n',layer_compression_rate)
    return model


def main():
    train_loader, test_loader = get_dataloader()
    if args.mode=='train':
        args.round=0
        model = ResNet18(num_classes=100)
        train_model(model, train_loader, test_loader)
    elif args.mode=='prune':
        previous_ckpt = 'resnet18-round%d.pth'%(args.round-1)
        print("Pruning round %d, load model from %s"%( args.round, previous_ckpt ))
        model = torch.load( previous_ckpt )
        params_ori = sum([np.prod(p.size()) for p in model.parameters()])
        print("Number of ori_Parameters: %.1fM"%(params_ori/1e6))
        #prune_model_with_shortcut(model)
        #prune_model_without_shortcut(model)
        prune_model_mixed(model)
        print(model)
        params = sum([np.prod(p.size()) for p in model.parameters()])
        print("Number of Parameters: %.1fM"%(params/1e6))
        train_model(model, train_loader, test_loader)

    elif args.mode=='test':
        ckpt = 'resnet18-round%d.pth'%(args.round)
        print("Load model from %s"%( ckpt ))
        model = torch.load( ckpt )
        params = sum([np.prod(p.size()) for p in model.parameters()])
        print("Number of Parameters: %.1fM"%(params/1e6))
        acc = eval(model, test_loader)
        print("Acc=%.4f\n"%(acc))

if __name__=='__main__':
    main()
