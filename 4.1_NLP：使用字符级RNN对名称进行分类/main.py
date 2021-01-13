# https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html

from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import unicodedata
import string
import tool


# print(findFiles("4.1_NLP：使用字符级RNN对名称进行分类/data/names/*.txt"))
all_letters = tool.get_All_Letters()
n_letters = len(all_letters)



# print(unicodeToAscii('Ślusàrski'))
# 每个文件的名称和内容
category_lines = {}
all_categories = []  # 所有文件名

for filename in tool.findFiles("4.1_NLP：使用字符级RNN对名称进行分类/data/names/*.txt"):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = tool.readLines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)

# print(category_lines['Italian'][:5])
