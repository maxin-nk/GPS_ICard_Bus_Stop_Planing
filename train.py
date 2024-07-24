# ! /usr/bin/env python
# coding:utf-8
# python interpreter:3.6.2
# author: admin_maxin
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import time
import random
from sklearn.cluster import KMeans
import math


# # =============1.绘制A市居民公交出行总人数折线图
# file_path = "data/time"
# file_names = os.listdir(file_path)
# labs = ["2017-6-9", "2017-6-10", "2017-6-11", "2017-6-12", "2017-6-13"]
# plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置显示字体
# plt.rcParams["axes.unicode_minus"] = False  # 设置正常显示符号
#
# for i in range(len(file_names)):  # 遍历文件数据
#     tmp = pd.read_csv(os.path.join(file_path, file_names[i]))
#     plt.plot(tmp["date"], tmp["num"], label=labs[i])
# # Places a legend on the axes.
# plt.legend()
#
# # plt.xticks: Get or set the current tick locations and labels of the x-axis
# plt.xticks([5, 7, 8, 10, 14, 18, 21, 23], ['5:00', '7:00', "8:00", '10:00', '14:00', '18:00', '21:00', '23:00'])
# plt.xlabel("时间")
# # np.linspace: Return evenly spaced numbers over a specified interval
# plt.yticks(list(np.linspace(100000, 500000, 5)), list(np.linspace(10, 50, 5)))
# plt.ylabel("人数（万人）")
#
# plt.title("2017/6/9-2017/6/13 A城市居民公交出行总情况")
#
# plt.show()


# ============2.抽取334路公交车GPS数据 / 刷卡数据

# # ============3.合并334路 -- GPS数据、刷卡数据
# szt_334 = pd.read_csv("data/stz_10_334.txt", sep="\t")  # shape:(20489, 7)
# gps_334 = pd.read_csv("data/gps_10_334.txt", sep="\t")  # shape:(214770, 14)
#
# current_time = time.time()
# a = list(szt_334.columns)
# a[-1] = "车牌号"  # 将“业务时间1” --> “车牌号”
# szt_334.columns = a
#
# # pandas.core.frame.DataFrame.value_counts(): Returns object containing counts of unique values
# License_plate_334 = szt_334["车牌号"].value_counts().index  # 获取334路所有车牌号
#
# data = pd.DataFrame()  # 新建存储334路经纬度数据的数据库
# for index in License_plate_334:
#     tmp_gps = gps_334.loc[gps_334["车牌号"] == index, ]  # index车牌对应的all gps数据
#     tmp_szt = szt_334.loc[szt_334["车牌号"] == index, ]  # index车牌对应的all szt数据
#
#     # 将字符串格式数据转换为时间格式
#     # pandas.to_datetime: Convert argument to datetime
#     tmp_gps["业务时间"] = pd.to_datetime(tmp_gps["业务时间"])
#     tmp_gps["记录时间"] = pd.to_datetime(tmp_gps["记录时间"])
#     # tmp_szt["交易时间"] = pd.to_datetime(tmp_szt["交易时间"])
#
#     # 刷卡数据新增两列
#     tmp_szt["经度"] = 0
#     tmp_szt["纬度"] = 0
#
#     # 刷卡数据：按照“业务时间”进行排序
#     tmp_gps.sort_values(by="业务时间", inplace=True)
#     for i in tmp_szt.index:
#
#         # 判断GPS数据是否延迟
#         ind = tmp_gps["业务时间"] < tmp_szt.loc[i, "交易时间"]
#         ind2 = ind.index[sum(ind)]
#         tmp_szt.loc[i, ["经度", "纬度"]] = tmp_gps.loc[ind2, ["经度", "纬度"]]
#         # for j in ind.index:
#         #     if False == ind[j]:
#         #         tmp_szt.loc[i, ['经度', '纬度']] = tmp_gps.loc[j, ['经度', '纬度']]  # 提取gps经纬度作为刷卡的位置信息
#         #         break  # 跳出循环
#
#     # Concatenate pandas objects along a particular axis with optional set logic along the other axes
#     data = pd.concat([data, tmp_szt])
#
# # 合并数据的运行时间：688.767786026001
# # print("合并数据的运行时间：%s 秒" % (time.time()-current_time))
# data.to_csv("334路公交经纬度数据.csv")



# # ============4.绘制刷卡数据经纬度散点图（合并后的数据）
# data = pd.read_csv("a.csv", index_col=0)
# plt.rcParams["font.sans-serif"] = ["SimHei"]
# plt.rcParams["axes.unicode_minus"] = False
# plt.rcParams['axes.titlesize'] = 18
#
# # plt.figure: Creates a new figure
# plt.figure(figsize=(12, 12))
#
# # 获取334路公交某一车牌号
# chepai1 = data["车牌号"].value_counts().index[8]
# chepai2 = data["车牌号"].value_counts().index[6]
# chepai3 = data["车牌号"].value_counts().index[3]
#
# # 取得当前车牌号对应的数据
# tmp1 = data.loc[data["车牌号"] == chepai1, ]
# tmp2 = data.loc[data["车牌号"] == chepai2, ]
# tmp3 = data.loc[data["车牌号"] == chepai3, ]
# plt.scatter(tmp1["经度"], tmp1["纬度"], color="blue", label="车牌1")
# plt.scatter(tmp2["经度"], tmp2["纬度"], color="red", label="车牌2")
# plt.scatter(tmp3["经度"], tmp3["纬度"], color="green", label="车牌3")
#
# plt.legend()
# plt.xlabel("经度", size=18)
# plt.ylabel("纬度", size=18)
# plt.title("A市334路公交部分车辆居民出行经纬散点图")
# plt.show()



# # ============5.乘客上车站点判断（聚类）
# #
# # ====解决问题：实际的公交车刷卡机反馈的经纬度与站点的经纬度是有一定差距的，故采用实际站点数据训练一个聚类模型，
# #              通过聚类模型预测实际的公交刷卡机返回的GPS数据属于哪一个站点的数据，从而计算出站点实际的上车人数
# #
# # ============①a.csv:334路公交 GPS（经度、纬度）和 SZT（刷卡记录） 合并的数据
# # ============②334BusStop:公交站点的真实数据（GPS经度、GPS纬度、关键词）
# # ============③设置与真实站点数相同的标签数 k=len(bus_stops)
# # ============④通过“站点真实数据（GPS经度、GPS纬度）”训练一个K-means聚类模型
# # ============⑤lab2stop记录“字典推导式”产生的{类别：站点名称}
# # ============⑥将用户刷卡数据（经度、纬度）放入聚类模型进行预测，获得聚类号，并转换为站点名称存入data["站点"]
# #
# # data["纬度"].notnull(): return a logical array of True or False
# # data.loc: urely label-location based indexer for selection by label
# data = pd.read_csv("a.csv", index_col=0)
# data = data.loc[data["纬度"].notnull(), ]
# bus_stops = pd.read_csv("data/334BusStop.csv", encoding="gbk")
# bus_stops = bus_stops[["GPS纬度", "GPS经度", "关键词"]]
#
# estimater = KMeans(n_clusters=len(bus_stops))  # 71 bus stops
# # KMeans().fit: compute k-means clustering
# # cluster by latitude and longitude
# estimater = estimater.fit(bus_stops[["GPS经度", "GPS纬度"]])
#
# # i: index
# # j: cluster label
# # bus_stop.iloc[i, 2]: 关键词（站点名称）
# # pd.DataFrame.iloc: Purely integer-location based indexing for selection by position.
# lab2stop = {j: bus_stops.iloc[i, 2] for i, j in enumerate(estimater.labels_)}  # {类别：站点名称}
# lab2stop2 = {j: i for i, j in enumerate(estimater.labels_)}  # {类别：站点序号}
#
# # 预测标签（数值）,并将预测数值便签转化为对应的站点名称（标签）
# # 转换成站点名称是为了更好的观察数据
# data["stop"] = estimater.predict(data[["经度", "纬度"]])
# data["站点"] = data["stop"].apply(lambda x: lab2stop[x])
# # 为了与预测数据进行对比
# data["true_stop"] = data["stop"].apply(lambda x: lab2stop2[x])
#
# f = data["stop"].value_counts()  # 各个站点预计上车人数统计
# #
# # pd.DataFrame.to_csv(): Write DataFrame to a comma-separated values (csv) file
# f.to_csv("b.csv")
#
# # 通过预测上车站点数据绘制散点图
# chepai = data['车牌号'].value_counts().index[3]
# tmp2 = data.loc[data['车牌号'] == chepai, ]
# plt.figure(figsize=(7, 7))
# plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置显示字体
# plt.rcParams["axes.unicode_minus"] = False  # 设置正常显示符号
#
# for i in range(len(bus_stops)):
#     tmp3 = tmp2.loc[tmp2['stop'] == i, ]
#     plt.scatter(tmp3['纬度'], tmp3['经度'])
#     plt.xlabel("纬度", size=12)
#     plt.ylabel("经度", size=12)
# plt.show()



# # ============6.上下行车路线拆分
# #
# # ============①获取去重的车牌号,及其对应数据
# # ============②判断“交易时间”的规范性 --> 公交是否休息（相邻两个时间做差）
# #
#
# # pandas.DataFrame.drop_duplicates(): Return DataFrame with duplicate rows removed, optionally only considering certain columns
# chepais = data["车牌号"].drop_duplicates()
#
# # 存储上线行数据
# data_sx = pd.DataFrame()
# data_xx = pd.DataFrame()
#
# for chepai in chepais:
#     # get current bus data
#     temp2 = data.loc[data["车牌号"] == chepai, ]
#     temp2["交易时间"] = pd.to_datetime(temp2["交易时间"])
#     temp2.sort_values(by="交易时间", inplace=True)
#     a = list(temp2["交易时间"])
#
#     # calculate the time difference between cards
#     # 相邻两个时间做差
#     b1 = pd.Series(a[:-1])
#     b2 = pd.Series(a[1:])
#     tim = b2-b1
#
#     # pandas.core.series.Series.astype: Cast to a NumPy array with 'dtype'.
#     # pandas.core.series.Series.apply: Invoke function on values of Series.
#     # 时间格式: "0 days 00:06:11"
#     # x[7:9]*60: 小时转换成分钟
#     # x[10:12]: 分钟
#     tim = tim.astype(str).apply(lambda x: int(x[6:9])*60 + int(x[10:12]))  # len(tim):732
#     tim = pd.Series([0] + list(tim))                                       # len(tim):733
#     tim.index = temp2.index
#     st_time = list(temp2.loc[tim > 30, "交易时间"])  # 时间差>30min认为公交休息
#     # 添加“公交车停止运营时间”
#     st_time = st_time + [temp2.loc[list(temp2["交易时间"].index)[-1], "交易时间"]]
#     # 添加“公交车开始运营时间”
#     st_time = st_time + [temp2.loc[list(temp2["交易时间"].index)[0], "交易时间"]]
#
#     a = st_time[::2]
#     b = st_time[1::2]
#     c = st_time[2::2]
#
#     # 获取实际站点数进行判断
#     # 凑运营时间段
#     if temp2.iloc[0, 11] < 20:
#         sx_time = [[a[i], b[i]] for i in range(min(len(a), len(b)))]
#         xx_time = [[b[i], c[i]] for i in range(min(len(b), len(c)))]
#     else:
#         xx_time = [[a[i], b[i]] for i in range(min(len(a), len(b)))]
#         sx_time = [[b[i], c[i]] for i in range(min(len(b), len(c)))]
#
#     # 提取上行路线的刷卡记录
#     for i in sx_time:
#         ind = temp2["交易时间"] >= i[0]
#         ind2 = temp2["交易时间"] < i[1]
#         temp3 = temp2.loc[ind & ind2, ]
#         data_sx = pd.concat([data_sx, temp3])
#
#     # 提取下行路线刷卡记录
#     for i in xx_time:
#         ind = temp2["交易时间"] >= i[0]
#         ind2 = temp2["交易时间"] < i[1]
#         temp3 = temp2.loc[ind & ind2, ]
#         data_xx = pd.concat([data_xx, temp3])
#
# # # 保存数据
# data_sx.to_csv("data_sx.csv")
# data_xx.to_csv("data_xx.csv")




# # ============7.上车人数统计
# # ============(334路公交所有车牌的上车人数)
# data_sx = pd.read_csv("data_sx.csv")
# data_xx = pd.read_csv("data_xx.csv")
# data_sx.sort_values(by="true_stop", inplace=True)
# data_xx.sort_values(by="true_stop", inplace=True)
# bus_stops = pd.read_csv("data/334BusStop.csv", encoding="gbk")
#
# num = len(bus_stops)
# a = list(data_sx["true_stop"])
# up_sx = np.array([[i, a.count(i)] for i in range(num)])
# b = list(data_xx["true_stop"])
# up_xx = np.array([[i, b.count(i)] for i in range(num)])
#
# # ============(334路公交 “某一车牌” 的上车人数)
# data_sx_one = data_sx.loc[data_sx["车牌号"] == "粤BH1177", ]
# data_xx_one = data_xx.loc[data_xx["车牌号"] == "粤BH1177", ]
# a = list(data_sx_one["true_stop"])
# up_sx = np.array([[i, a.count(i)] for i in range(num)])
# b = list(data_xx_one["true_stop"])
# up_xx = np.array([[i, b.count(i)] for i in range(num)])
#
# print(up_sx, up_xx)
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
# plt.rcParams['axes.titlesize'] = 10
# plt.xticks([0, 2, 5, 9, 12, 18, 21, 24, 28, 34, 39, 45, 50, 54, 58, 65])
# plt.yticks([1, 5, 10, 15, 20, 25, 30, 40])
# plt.xlabel("站点")
# plt.ylabel("上车人数")
# plt.title("A市 - 334路公交某一站牌上下行站点上车人数折线图")
# plt.plot(up_sx[:, 1], color="red", label="上行线路")
# plt.plot(up_xx[:, 1], color="blue", label="下行线路")
# plt.legend()
# # plt.show()




# ============8.下车人数统计
data_sx = pd.read_csv("data_sx.csv", encoding="gbk")
data_xx = pd.read_csv("data_xx.csv", encoding="gbk")
bus_stops = pd.read_csv("data/334BusStop.csv", encoding="gbk")
bus_stops = bus_stops[['GPS纬度', 'GPS经度', '关键词']]

num = len(bus_stops)
a = list(data_sx["true_stop"])
up_sx = [a.count(i) for i in range(num)]
a = list(data_xx["true_stop"])
up_xx = [a.count(i) for i in range(num)]


# ============No.1 计算data_sx下车人数
# ①吸引权重
W_xy = np.array(up_sx)/sum(up_sx)

# ②泊松分布（计算OD概率矩阵）
F_bs = np.zeros((num, num))
lamd = np.floor(num/2)
for j in range(num):
    for i in range(0, j):
        F_bs[i, j] = (np.e**(-lamd) * lamd**(j-i))/math.factorial(j-i)

# ③下车人数计算
od_sx = np.zeros((num, num))

for j in range(num):
    a = sum(F_bs[j, :] * W_xy[:])
    for k in range(j, num):
        od_sx[j, k] = (F_bs[j, k] * W_xy[k]) / a

OD_sx = np.zeros((num, num))
for j in range(num):
    for k in range(j+1, num):
        OD_sx[j, k] = sum(od_sx[j, :k] * up_sx[:k])

f1 = pd.DataFrame(OD_sx)
f2 = pd.DataFrame(up_sx)
f1.to_csv("OD_sx.csv")
f2.to_csv("up_sx.csv")




# # ============No.2 计算data_xx下车人数
# # ①权重
# W_xy_xx = np.array(up_sx)/sum(up_sx)
#
# # ②泊松分布
# F_bs_xx = np.zeros((num, num))
# lamd = np.floor(num/2)
# for j in range(num):
#     for i in range(0, j):
#         F_bs_xx[i, j] = (np.e**(-lamd) * lamd**(j-i))/math.factorial(j-i)
#
# # ③下车人数计算
# od_xx = np.zeros((num, num))
#
# for j in range(num):
#     a = sum(F_bs_xx[j, :] * W_xy_xx[:])
#     for k in range(j, num):
#         od_xx[j, k] = (F_bs_xx[j, k] * W_xy_xx[k]) / a
#
# # 测试行：看概率是否 <1
# # f = pd.Series(F_bs[1, :]).sum()
# # g = pd.Series(od_sx[60, :]).sum()
#
# OD_xx = np.zeros((num, num))
# for j in range(num):
#     for k in range(j+1, num):
#         OD_xx[j, k] = sum(od_xx[j, :k] * up_xx[:k])
#
#
# f = pd.DataFrame(OD_sx)
# f.to_csv("OD_sx.csv")
# f = pd.DataFrame(OD_xx)
# f.to_csv("OD_xx.csv")






































