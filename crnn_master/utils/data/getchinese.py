def get_chinese(root):
    res = []
    with open(root, encoding='utf-8') as f:
        lines = f.readlines()
        f.close()

    i = 0
    for line in lines:
        chinese = line.strip().split('.jpg ')[1]
        i += 1
        res.append(chinese)
    # 字符集保存位置
    with open('../../data/formula.txt', 'w+', encoding='utf-8') as f:
        string = ''.join(res)
        string = set(string)
        string = list(string)
        string.sort()
        string = ''.join(string)
        f.write(string)


if __name__ == '__main__':
    # 标签位置（自己改）
    root = '/Users/loneranger/deep_learning/homework/Final_Project/crnn-master/data/labels.txt'
    get_chinese(root)
