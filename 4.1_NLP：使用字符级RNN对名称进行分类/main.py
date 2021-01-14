# https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html

from __future__ import unicode_literals, print_function, division
from io import open
import torch
import torch.nn as nn
import glob
import os
import unicodedata
import string
import sys
import time
import math
sys.path.append("./")  # 将同级目录加入系统路径，方便导入自定义模块
import NeuralNet
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
# print(tool.letter2Tensor('J'))

# print(tool.line2Tensor('Jones').size())


n_hidden = 128
rnn = NeuralNet.RNN(n_letters, n_hidden, n_categories)
loss_fn = nn.NLLLoss()
# learning_rate=0.005
optim = torch.optim.ASGD(rnn.parameters(), 0.005)


def train(category_tensor, line_tensor):
    hidden = rnn.initHidden()
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)
    loss = loss_fn(output, category_tensor)
    optim.zero_grad()
    loss.backward()
    optim.step()
    return output, loss.item()


n_iters = 10000
print_every = 5000
plot_every = 1000

current_loss = 0
all_losses=[]


def timeSince(since):
    now = time.time()
    s = now-since
    m = math.floor(s/60)
    s -= m*60
    return "%dm %ds" % (m, s)


start = time.time()

for iter in range(1, n_iters+1):
    category, line, category_tensor, line_tensor = tool.randomTrainingExample(all_categories, category_lines)
    output,loss=train(category_tensor,line_tensor)
    current_loss+=loss

    if iter %print_every==0:
        guess,guess_i=tool.categoryFromOutput(output,all_categories)
        correct="√"if guess==category else "✗(%s)"%category
        print("%d %d%% (%s) %.4f %s / %s %s"%(iter,iter/n_iters*100,timeSince(start),loss,line,guess,correct))

    if iter %plot_every==0:
        all_losses.append(current_loss/plot_every)
        current_loss=0


# input = tool.line2Tensor("Albert")
# hidden = torch.zeros(1, n_hidden)
# output, next_hidden = rnn(input[0], hidden)
# print(tool.categoryFromOutput(output,all_categories))

# for i in range(10):
#     category, line, category_tensor, line_tensor = tool.randomTrainingExample(all_categories,category_lines)
#     print('category =', category, '/ line =', line)



