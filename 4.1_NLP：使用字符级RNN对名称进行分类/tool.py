from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import unicodedata
import string


def get_All_Letters():
    '''返回ascii字符集'''
    return string.ascii_letters+".,;'"


def findFiles(path):
    '''返回路径下所有符合条件的文件'''
    return glob.glob(path)


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
        and c in get_All_Letters()
    )


def readLines(filename):
    lines = open(filename, encoding="utf-8").read().strip().split("\n")
    return [unicodeToAscii(line)for line in lines]
