#! /usr/bin/env python3
# -*-coding: utf-8-*-

"""
Created on Apr 12

@author: Moonkie

@attention: 
     从jaffe人脸库获取训练数据和标签
"""

import glob
import numpy as np
import os


CURDIR = os.path.split(os.path.abspath(__file__))[0]
ROOT = os.path.split(CURDIR)[0]
imgdir = os.path.join(CURDIR,'trains','jaffe')
lst = os.path.join(CURDIR,'train.lst')
lblst = os.path.join(CURDIR,'trains','jaffe.csv')
validatfile = os.path.join(CURDIR,'data','validate.log')
kp = os.path.join(imgdir,'*[.tiff|.jpg|.png|.bmp]')

def emo_label(data):
    '''

    data structure like that '[x,y,z...]'
    '''
    # assert len(data) == 1,"Just need 1-dimension array"
    data = np.array(data,np.float)
    if np.max(data) - np.mean(data) < 1:
        return 6
    # return (data - np.min(data))/(np.ptp(data))
    return np.argmax(data)
        

lbls = np.loadtxt(lblst,bytes,delimiter=',',usecols=(0,1,2,3,4,5,6,7))
rsls = []

# emotion = [0,1,2,3,4,5,6]
validate_log_file = open(validatfile,'w+',encoding='utf8')
validate_log_file.write("index\temotion\n")
for lbl in lbls:
    idx = int(lbl[0])
    filename = lbl[-1].decode('utf8').replace('-','.')
    filename = '.'.join([filename,str(idx),'tiff'])
    filename = os.path.join(imgdir,filename)
    emo = emo_label(lbl[1:-1]) # np.array(lbl[1:],np.float)
    if os.path.exists(filename):
        print("%d %d %s" % (idx,emo,filename))
        rsls.append((idx,emo,filename))
        validate_log_file.write("%d\t\t%d\n" % (idx,emo))
        validate_log_file.flush()
    #     print('%d  %s  %s' % (idx," ".join([str("%.6f" % e) for e in emos]),filename))
    #     # rsls.append(([0. if ("%.3f" % e)==0 else e for e in emos],filename))
    #     rsls.append((emos,filename))
validate_log_file.close()

rsls = np.array(rsls)
np.savez(lst,label_file=rsls)
print('[^_^] Create train.lst successfully.')

