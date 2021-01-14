from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import unicodedata
import string
import torch
import random


def get_All_Letters():
    '''返回ascii字符集'''
    return string.ascii_letters+".,;'"


def findFiles(path):
    '''返回路径下所有符合条件的文件'''
    return glob.glob(path)


def unicodeToAscii(s):
    '''将unicode编码转换为ascii编码'''
    return ''.join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
        and c in get_All_Letters()
    )


def readLines(filename):
    '''读取文件内容，按换行符拆分
    Args:
        filename: 文件路径.
    Returns:
        返回List数组
    '''
    lines = open(filename, encoding="utf-8").read().strip().split("\n")
    return [unicodeToAscii(line)for line in lines]


def letter2Index(letter):
    '''计算字母的索引'''
    return get_All_Letters().find(letter)


def letter2Tensor(letter):
    '''将字母转换为Tensor

    例如：“b”=[[0,1,0,0,...]]

    例如：“bc”=[[0,1,0,0,...],[0,0,1,0,...]]
    '''
    tensor = torch.zeros(1, len(get_All_Letters()))
    tensor[0][letter2Index(letter)] = 1
    return tensor


def line2Tensor(line):
    '''将一行字母（一个完整单词）转换为Tensor

    例如：“b”=[[0,1,0,0,...]]

    例如：“bc”=[[0,1,0,0,...],[0,0,1,0,...]]
    '''
    tenosr = torch.zeros(len(line), 1, len(get_All_Letters()))
    for index, letter in enumerate(line):
        tenosr[index][0][letter2Index(letter)] = 1
    return tenosr


def categoryFromOutput(output, categories):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return categories[category_i], category_i


def randomChoice(l):
    '''从给定数组中随机选中一个'''
    return l[random.randint(0, len(l)-1)]


def randomTrainingExample(all_categories, category_lines):
    '''生成训练所需的随机数据'''
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = torch.tensor(
        [all_categories.index(category)], dtype=torch.long)
    line_tensor = line2Tensor(line)
    return category, line, category_tensor, line_tensor



