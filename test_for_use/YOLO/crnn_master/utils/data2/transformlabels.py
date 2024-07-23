with open('../../data/char_std_5990.txt', encoding='utf-8') as f:
    content = f.read()
    content = content.replace('\n', '')
    f.close()


def transform_labels():
    data = []
    with open('../../data/data_test.txt', encoding='utf-8') as f:
        c_list = f.readlines()
        for c in c_list:
            image = c.split(' ')[0]
            label = c.split(' ')[1:]
            new_label = [content[int(i)] for i in label]
            new_label = ''.join(new_label)
            new_label.replace('︰', ':').replace('﹐', ',').replace('﹑', '')
            data.append(image + ' ' + new_label)

    with open('../../data/data_train.txt', encoding='utf-8') as f:
        c_list = f.readlines()
        for c in c_list:
            image = c.split(' ')[0]
            label = c.split(' ')[1:]
            new_label = [content[int(i)] for i in label]
            new_label = ''.join(new_label)
            data.append(image + ' ' + new_label)

    data.sort()
    data = '\n'.join(data)
    with open('../../data/labels.txt', 'w+', encoding='utf-8') as f:
        f.write(data)


if __name__ == '__main__':
    transform_labels()
