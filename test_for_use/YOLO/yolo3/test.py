import sys
import os
import time
from plotly.offline import iplot
from streamlit import cli as stcli
import streamlit as st
import numpy as np
from PIL import Image, ImageFont, ImageDraw
import matplotlib.pyplot as plt
from yolo import YOLO
from crnn_master.hw4_2 import parse_opt, main
import shutil
from calculate import outcome
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

acc = go.Scatter(
        x=['训练集', '验证集', '测试集'],
        y=[0.9835243553008596, 0.9664634146341463, 0.9024390243902439],
        name='acc'
    )
    #loss
loss = go.Bar(
        x=['训练集', '验证集', '测试集'],
        y=[0.8488874537896973, 0.5683902647437119, 0.5204568483480593],
        name='loss'
    )

fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(acc)
fig.add_trace(loss, secondary_y=True)
fig['layout'].update(height=600, width=800, title='不同数据集准确率和损失对比图', xaxis=dict(
    tickangle=-90
))
iplot(fig)
# st.subheader("Detection accuracy analysis")
# st.plotly_chart(fig)