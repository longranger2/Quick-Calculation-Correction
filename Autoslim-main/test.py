import torch_pruning as pruning
from torchvision.models import resnet18
import torch

#模型建立
model=resnet18()

#剪枝引擎建立
slim=pruning.Autoslim(model,inputs=torch.randn(1,3,224,224),compression_ratio=0.5)

#剪枝，系统默认prune_shortcut=1,prune_shortcut=0时不剪跳连层
slim.l1_norm_pruning(prune_shortcut=1)

#使用自己的train函数，对剪枝后的model微调
train(model)

#微调后，保存模型
torch.save(model,'./resnet18.pth')

#如果需要再次载入模型，使用下面的语句
resnet18_pruned_model=torch.load('./resnet18.pth')