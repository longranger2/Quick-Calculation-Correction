# 部署

## 环境配置
1. 配置环境
```bash
# 创建环境
conda create -n dlhomework python=3.8
# 激活环境
conda activate dlhomework

# 安装torch 1.7.1
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch

# 安装streamlit等一些列的包 使用阿里云镜像加速下载
pip install streamlit opencv-python plotly matplotlib -i https://mirrors.aliyun.com/pypi/simple/

# 可能会遇到的问题
# Pillow报错
pip install --upgrade Pillow -i https://mirrors.aliyun.com/pypi/simple/
# matplotlib不正确 （没安装上）
pip install matplotlib -i https://mirrors.aliyun.com/pypi/simple/
```
2. 下载模型文件
分别放入 yolo3/model_data 和 cnn_master/weight 下

## 运行
```bash
# 激活环境
conda activate dlhomework
# 运行代码
python demo.py
```
