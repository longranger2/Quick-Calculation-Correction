import torch.nn as nn


class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        """

        :param nIn: 输入层神经元个数
        :param nHidden: 隐藏层神经元个数
        :param nOut: 输出层神经元个数
        """
        super(BidirectionalLSTM, self).__init__()
        # 双向LSTM
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        # 两个方向的隐藏层单元频在一起，所以nHidden*2
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        # T:时间序列 b:batch_size h:隐藏层神经元
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)
        output = self.embedding(t_rec)
        output = output.view(T, b, -1)
        return output


class CRNN(nn.Module):

    def __init__(self, imgH, nc, nclass, nh):
        """

        :param imgH: 图片高度
        :param nc: 图片通道数
        :param nclass: 类别个数
        :param nh: RNN中隐藏层神经元个数
        """
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, '图片高度必须是16的倍数，建议32'

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=nc, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 1), padding=(0, 1)),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 1), padding=(0, 1)),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=2, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
        )

        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nclass)
        )

    def forward(self, input):
        conv = self.cnn(input)
        # print(conv.size())

        b, c, h, w = conv.size()
        assert h == 1, '图片高度经过卷积之后必须为1'
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)

        output = self.rnn(conv)
        # print(output.size())
        return output
