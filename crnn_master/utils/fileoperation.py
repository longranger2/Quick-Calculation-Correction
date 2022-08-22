import os
import sys

def get_chinese(path):
    cur_path = os.path.abspath(os.path.dirname(__file__))
    print(cur_path)
    with open(path, 'r', encoding='utf-8') as f:
        chinese = f.read()
        f.close()
        return chinese


if __name__ == '__main__':
    #f=open('../data/formula.txt','r')
    path = '../data/formula.txt'
    with open(path, 'r', encoding='utf-8') as f:
        print(f)


