# ! /usr/bin/env python
# coding:utf-8
# python interpreter:3.6.2
# author: admin_maxin
import pandas as pd
import matplotlib.pyplot as plt
import random
import numpy as np

b = pd.read_csv("b.csv")
b.sort_index(by="站牌", inplace=True)
print(b.shape)

xlabels = list(b["站牌"])
data = list(b["上车人数"])

plt.figure(figsize=(7, 7))
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置显示字体
plt.rcParams["axes.unicode_minus"] = False  # 设置正常显示符号
plt.xlabel("站点", size=12)
plt.ylabel("人数/个", size=12)
plt.plot(xlabels, data, color="g")
plt.show()



