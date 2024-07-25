import torch
import collections


class StrLabelConverter(object):

    def __init__(self, chinese):
        self.chinese = chinese + '@'
        self.dict = {}
        for i, char in enumerate(self.chinese):
            self.dict[char] = i + 1

    def encode(self, text):
        """

        :param text: text可以是字符串，也可以是列表
        :return: tensor格式的编码后的中文和长度
        """
        if isinstance(text, str):
            text = [self.dict[char] for char in text]
            length = [len(text)]
        elif isinstance(text, collections.Iterable):
            length = [len(s) for s in text]
            text = ''.join(text)
            text, _ = self.encode(text)
        return torch.IntTensor(text), torch.IntTensor(length)

    def decode(self, t, length, raw=False):
        """

        :param t: 预测中文的列表
        :param length: 中文的长度
        :param raw: False的话就直接转换(含有@)，否则返回不含-的中文
        :return: 中文的字符串
        """
        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length
            if raw:
                return ''.join([self.chinese[i - 1] for i in t])
            else:
                chinese_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        chinese_list.append(self.chinese[t[i] - 1])
                return ''.join(chinese_list)
        else:
            assert t.numel() == length.sum()
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(t[index:index + l], torch.IntTensor([l]), raw=raw)
                )
                index += l
            return texts