import torch
import torch.nn as nn
import argparse
from crnn import CRNN
import time
import tqdm
from utils.aftertreatment import StrLabelConverter
from utils.fileoperation import get_chinese
from nni.algorithms.compression.pytorch.pruning import LevelPruner
from nni.algorithms.compression.pytorch.quantization import QAT_Quantizer
from nni.compression.pytorch import ModelSpeedup


chinese = get_chinese('data/formula.txt')
converter = StrLabelConverter(chinese)
nclass = len(chinese) + 1
crnn = CRNN(32, 1, nclass, 256)

# if os.path.exists(opt.weights):
#     crnn.load_state_dict(torch.load(opt.weights))
# log_load_model(opt.weights)
#
# optimizer = torch.optim.Adam(crnn.parameters(), lr=0.001)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# crnn = crnn.to(device)


# model: 要加速的模型
# dummy_input: 模型的示例输入，传给 `jit.trace`
# masks_file: 剪枝算法创建的掩码文件
print(crnn)
m_speedup = ModelSpeedup(crnn, dummy_input=torch.rand(4,1,32,32), masks_file='weights/mask_best_1_6.429664146609422_0.8689024390243902.pt')
#m_speedup = ModelSpeedup(crnn, dummy_input=torch.rand(32,1,256,256), masks_file='weights/mask_best_1_6.429664146609422_0.8689024390243902.pt')
m_speedup.speedup_model()
print(crnn)
# evaluation_result = evaluator(model)
# print('Evaluation result (speed up model): %s' % evaluation_result)
# result['performance']['speedup'] = evaluation_result

torch.save(crnn.state_dict(),'./weights/model_speed_up.pth')
print('Speed up model saved to weights/model_speed_up.pth')
# dummy_input = dummy_input.to(device)
# start = time.time()
# out = crnn(dummy_input)
# print('elapsed time: ', time.time() - start)
