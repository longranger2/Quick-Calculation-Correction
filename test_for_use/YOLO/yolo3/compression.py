import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.yolo import YoloBody
from nets.yolo_training import YOLOLoss, weights_init
from utils.callbacks import LossHistory
from utils.dataloader import YoloDataset, yolo_dataset_collate
from utils.utils import get_anchors, get_classes
from utils.utils_fit import fit_one_epoch

import torch
from pytorchyolo import models

from nni.compression.pytorch import ModelSpeedup
from nni.algorithms.compression.pytorch.pruning import L1FilterPruner, LevelPruner
from nni.compression.pytorch.utils import not_safe_to_prune
from nni.algorithms.compression.pytorch.quantization import QAT_Quantizer

if __name__ == "__main__":
    # -------------------------------#
    #   是否使用Cuda
    #   没有GPU可以设置成False
    # -------------------------------#
    Cuda = False
    # --------------------------------------------------------#
    #   训练前一定要修改classes_path，使其对应自己的数据集
    # --------------------------------------------------------#
    classes_path = 'model_data/my_classes.txt'
    # ---------------------------------------------------------------------#
    #   anchors_path代表先验框对应的txt文件，一般不修改。
    #   anchors_mask用于帮助代码找到对应的先验框，一般不修改。
    # ---------------------------------------------------------------------#
    anchors_path = 'model_data/yolo_anchors.txt'
    anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    #导入初始模型参数
    model_path = 'model_data/yolo_weights.pth'
    #   获取classes
    class_names, num_classes = get_classes(classes_path)
    # ------------------------------------------------------#
    #   创建yolo模型
    # ------------------------------------------------------#
    #初始模型
    model = YoloBody(anchors_mask, num_classes)
    weights_init(model)

    #剪枝模型
    # config_list = [{
    #     'sparsity': 0.5,
    #     'op_types': ['default'],
    # }]
    # model = YoloBody(anchors_mask, num_classes)
    # weights_init(model)
    # pruner = LevelPruner(model, config_list)
    # model = pruner.compress()

    #模型量化
    # config_list = [{
    #     'quant_types': ['weight'],
    #     'quant_bits': {
    #         'weight': 8,
    #     },  # 这里可以仅使用 `int`，因为所有 `quan_types` 使用了一样的位长，参考下方 `ReLu6` 配置。
    #     'op_types': ['Conv2d', 'Linear']
    # }, {
    #     'quant_types': ['output'],
    #     'quant_bits': 8,
    #     'quant_start_step': 7000,
    #     'op_types': ['ReLU6']
    # }]
    #
    # model = YoloBody(anchors_mask, num_classes)
    # weights_init(model)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    # quantizer = QAT_Quantizer(model, config_list,optimizer)
    # quantizer.compress()

    if model_path != '':
        print('Load weights {}.'.format(model_path))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    #模型压缩
    print(model)
    model.eval()
    dummy_input = torch.rand(8, 3, 320, 320)
    model(dummy_input)
    # Generate the config list for pruner
    # Filter the layers that may not be able to prune
    not_safe = not_safe_to_prune(model, dummy_input)
    cfg_list = []
    for name, module in model.named_modules():
        if name in not_safe:
            continue
        if isinstance(module, torch.nn.Conv2d):
            cfg_list.append({'op_types': ['Conv2d'], 'sparsity': 0.6, 'op_names': [name]})
    # Prune the model
    pruner = L1FilterPruner(model, cfg_list)
    pruner.compress()
    pruner.export_model('./compression/model.pth', './compression/mask.pth')
    pruner._unwrap_model()
    # Speedup the model
    ms = ModelSpeedup(model, dummy_input, './compression/mask.pth')
    ms.speedup_model()
    model(dummy_input)
    print(model)
    torch.save(model, './compression/YOLO.pth')

