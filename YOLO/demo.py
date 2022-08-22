import sys
import os
import time
from streamlit import cli as stcli
import streamlit as st
import numpy as np
import torch
from PIL import Image, ImageFont, ImageDraw
import matplotlib.pyplot as plt
from yolo import YOLO
from crnn_master.hw4_2 import parse_opt, main
import shutil
from calculate import outcome
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import cv2 as cv


def Labeling():
    st.write("labeling!!!")


def ModelUpdate():
    st.write('ModelUpdate!!!')


def get4pos(box, image):
    top, left, bottom, right = box
    top = max(0, np.floor(top).astype('int32'))
    left = max(0, np.floor(left).astype('int32'))
    bottom = min(image.size[1], np.floor(bottom).astype('int32'))
    right = min(image.size[0], np.floor(right).astype('int32'))
    return top, left, bottom, right


# 返回yolo框出的区域，并将其等式图片存入对应文件夹中
def GetBoxesPic(image, boxes):
    pics = []
    shutil.rmtree('./yolo3/tmp_img')  # 清空操作
    os.mkdir('./yolo3/tmp_img')
    for i in range(len(boxes)):
        top, left, bottom, right = get4pos(boxes[i], image)
        pic = image.crop((left - 15, top, right + 40, bottom))
        pic.save('./yolo3/tmp_img/pic' + str(i).rjust(3, '0') + '.jpg')
        pics.append(pic)
    return pics


# 进行yolo检测，呈现在web页面上
def Detecting(image):
    st.subheader("Detected Image")
    st.write("Just a second ...")
    yolo = YOLO()
    #加载压缩后的YOLO模型
    #yolo = torch.load("/Users/loneranger/deep_learning/hoFinal_Project/YOLO/yolo3/compression/YOLO.pth")
    my_bar = st.progress(0)
    img = image.copy()
    start1 = time.time()
    r_image, boxes, top_conf = yolo.detect_image(image)
    end1 = time.time()
    # print(boxes)
    for percent_complete in range(100):
        my_bar.progress(percent_complete + 1)
    st.image(r_image, use_column_width=True)  # 展现检测结果
    # st.download_button(label="Download image", data=r_image, file_name='large_df.jpg', mime="image/jpg")
    st.subheader("Detection outcome Analysis")
    plt.scatter(np.arange(len(top_conf)), top_conf)
    plt.xlabel('detected rectangle')
    plt.ylabel('score')
    st.pyplot()
    # st.balloons()
    pics = GetBoxesPic(img, boxes)
    return boxes, start1, end1
    # st.image(pic, use_column_width=True)


def painting(equations, image, boxes):
    st.subheader("Identification outcome")
    imgdraw = ImageDraw.ImageDraw(image)  # 创建一个绘图对象，传入img表示对img进行绘图操作
    font = ImageFont.truetype('Microsoft Sans Serif.ttf', image.size[1] // 50, encoding="utf-8")
    for i in range(len(boxes)):
        top, left, bottom, right = get4pos(boxes[i], image)
        if outcome(equations[i]):
            imgdraw.text(xy=(left, bottom + 3), text=equations[i] + '√', fill=(255, 0, 0), font=font)
        else:
            imgdraw.text(xy=(left, bottom + 3), text=equations[i] + '×', fill=(255, 0, 0),
                         font=font)  # 调用绘图对象中的text方法表示写入文字
    st.image(image, use_column_width=True)


def detec_acc_analysis():
    # train_loss=0.8488874537896973, train_acc=0.9835243553008596
    # val_loss=0.5683902647437119, val_acc=0.9664634146341463
    # test_loss=0.5204568483480593, test_acc=0.9024390243902439

    # # 柱状簇
    # # Trace
    # #acc
    # acc = go.Bar(
    #     x=['训练集','验证集', '测试集'],
    #     y=[0.9835243553008596, 0.9664634146341463, 0.9024390243902439],
    #     name='acc'
    # )
    # #loss
    # loss= go.Bar(
    #     x=['训练集','验证集', '测试集'],
    #     y=[0.8488874537896973,0.5683902647437119,0.5204568483480593],
    #     name='loss'
    # )
    # trace = [acc, loss]
    # #Layout
    # layout = go.Layout(
    #     title='不同数据集准确率和损失对比图'
    # )
    # # Figure
    # fig = go.Figure(data=trace, layout=layout)
    # st.plotly_chart(fig)

    acc = go.Scatter(
        x=['训练集', '验证集', '测试集'],
        y=[0.9835243553008596, 0.9664634146341463, 0.9024390243902439],
        name='acc'
    )
    # loss
    loss = go.Bar(
        x=['训练集', '验证集', '测试集'],
        y=[0.8488874537896973, 0.5683902647437119, 0.5204568483480593],
        name='loss'
    )
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(loss)
    fig.add_trace(acc, secondary_y=True)
    fig['layout'].update(height=600, width=800, title='不同数据集准确率及损失对比图')
    st.subheader("Detection accuracy analysis")
    st.plotly_chart(fig)


def acc_analysis(equations, image, boxes):
    sum = len(boxes)
    count = 0
    for i in range(len(boxes)):
        top, left, bottom, right = get4pos(boxes[i], image)
        if outcome(equations[i]):
            count += 1
    st.subheader("Answer accuracy analysis")
    labels = ['答对数', '答错数']
    values = [count, sum - count]
    fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
    # fig.show()
    st.plotly_chart(fig)


def Time(detec_time, recog_time):
    st.subheader("Reasoning time analysis")
    labels = ['Time']
    fig = go.Figure(data=[
        go.Bar(name='检测时间', x=labels, y=[detec_time]),
        go.Bar(name='识别时间', x=labels, y=[recog_time])
    ])
    # fig.show()
    st.plotly_chart(fig)


def tt():
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.title("Handwriting Recognition")
    st.write("")
    file_up = st.file_uploader("Upload an image", type="jpg")

    if file_up is not None:
        image = Image.open(file_up)
        st.subheader("Uploaded Image")
        st.image(image, use_column_width=True)
        st.write("")
        img = image.copy()
        if st.button('Submit it'):
            st.write("Succeed!!!")
        if st.button('Labeling'):
            Labeling()
        if st.button('Model Update'):
            ModelUpdate()
        if st.button('Detecting'):
            st.subheader("Recognition")
            # 等式检测
            boxes, start1, end1 = Detecting(image)
            # 文本识别
            start2 = time.time()
            opt = parse_opt()
            equations = main(opt)
            end2 = time.time()
            painting(equations, img, boxes)
            # 高级要求
            # 文本识别率识别分析
            detec_acc_analysis()
            # 用户答题准确率分析
            acc_analysis(equations, img, boxes)
            # 推理时间分析
            Time(end1 - start1, end2 - start2)


if __name__ == '__main__':

    if st._is_running_with_streamlit:
        tt()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
