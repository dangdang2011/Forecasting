# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as np
import  os
import sys

#声明一个字典类型，备用
d = {}
f = open('./dataSet/video_create_log.txt')
fo = open('./dataSet/video_create_pre.csv', "w")
#读文件并且处理的操作
while True:
    #按行读取文件
    s = f.readline()
    #如果文件读完了，退出
    if s == "":
        break
    #每一行按照你\t分隔，分别存在a和b中
    a, b = map(int, s.split('\t'))
    #如果字典中已经a，数目加1，如果没有，数目初始化为1
    if a in d:
        d[a] += 1
    else:
        d[a] = 1
#安装key排序
items = d.items()
items.sort()
#将每一行写入文件pre_video_create.csv中
for key,value in items:
    fo.write( str(key) + "," + str(value) + "\n")
fo.close()
f.close()