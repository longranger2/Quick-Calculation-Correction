# 如何训练自己的数据集呢？
https://github.com/zhijiezhong/crnn/tree/master
- 从网上找数据集下载下来
- 图片标签格式为：图片名字 标签，如果不是这种格式的话，自己转换一下就可以了
![img_1.png](img/img_1.png)
  
- 我们需要训练集，验证集和测试集的标签，还有字符集
- 想要获得字符集可以运行utils/data/getchinese.py
![img_2.png](img/img_2.png)
  
- 想要训练集，验证集和测试集的标签可以运行utils/data/splitlabels.py
![img_3.png](img/img_3.png)
  
- 打开train.py，根据需求更改参数，然后右键运行，模型就开始训练了
![img_4.png](img/img_4.png)
  
- 打开test.py，可以测试模型的效果，修改一下参数，再测试
![img_5.png](img/img_5.png)
![img_6.png](img/img_6.png)
  
- 打开detect.py,可以对图片进行推理，修改一下参数，再推理
![img_7.png](img/img_7.png)
![img_8.png](img/img_8.png)
**训练自己的模型就是这么简单！**
  
  
