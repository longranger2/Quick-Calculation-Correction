import tqdm

ftrain = open('../../data/labels/train.txt', 'w+', encoding='utf-8')
fval = open('../../data/labels/val.txt', 'w+', encoding='utf-8')
ftest = open('../../data/labels/test.txt', 'w+', encoding='utf-8')


def split_labels():
    with open('../../data/labels.txt', encoding='utf-8') as f:
        labels = f.readlines()
    num = len(labels)

    i = 0
    for label in tqdm.tqdm(labels):
        if i < num * 0.75:
            ftrain.write(label)
        elif i < num * 0.95:
            fval.write(label)
        else:
            ftest.write(label)
        i += 1


if __name__ == '__main__':
    split_labels()

