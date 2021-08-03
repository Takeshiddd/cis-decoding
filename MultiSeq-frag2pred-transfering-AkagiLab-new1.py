#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Inputは「改行を削除した」fasta。一つのモデル（h5またはhdf5）に対して、
# 各seqをbin-sizeにwalkingで断片化し、sequentialに結合predictionを出力

# outputは各行に連続的な結合prediction（positiveの確率）を出力。wOTUも同時に出力


# In[ ]:


import numpy as np
import os
import re
import numpy as np
import tensorflow as tf
import keras
#from keras.layers import (Activation, Add, GlobalAveragePooling2D,
#                          BatchNormalization, Conv1D, Conv2D, Dense, Flatten, Reshape, Input, Dropout,
#                          MaxPooling1D,MaxPooling2D)
#from keras.models import Sequential, Model
#from keras.layers import Dense, GlobalAveragePooling2D
from keras import optimizers
from keras import backend as K
#from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras.utils import plot_model,np_utils
#from keras.regularizers import l2
#from sklearn.model_selection import train_test_split
#from sklearn.utils.class_weight import compute_class_weight
#from sklearn.metrics import confusion_matrix
from keras.models import Model, load_model
#from functools import reduce
#from sklearn.metrics import classification_report, confusion_matrix


# In[ ]:


from optparse import OptionParser
usage = "USAGE: this.py [-f] [-m] [-w] [-b] [-o]"
parser = OptionParser(usage=usage)
parser.add_option("-f", dest="fasta", action="store",help="File path to fasta")
parser.add_option("-m", dest="trainmodel", action="store", help="File path to HDF5 or H5 model")
parser.add_option("-w", dest="walk", action="store", help="walk bp size")
parser.add_option("-b", dest="bin", action="store", help="bin bp size")
parser.add_option("-o", dest="out", action="store", help="output file name")


(opt, args) = parser.parse_args()

fasta = opt.fasta
trainmodel = opt.trainmodel
walk = opt.walk
bin = opt.bin
out = opt.out

# 下準備1：OTU/seqを分離 それぞれprepped-OTU.txt, prepped-seq.faとして保存
os.system("awk '!/>/ {print}' " + fasta + " > prepped-seq.fa")
os.system("awk '/>/ {print}' " + fasta + " > prepped-OTU.txt")
          
outfile =open(out, "w")
fas = open("prepped-seq.fa",'r')
fas1 = fas.readlines()


# In[ ]:


# 下準備2: シークエンスのアレイ化コード

def dna2num(dna): # DNA配列を数値に変換。nt2intみたいな。1文字ずつ
    if dna.upper() == "A":
        return 0
    elif dna.upper() == "T":
        return 1
    elif dna.upper() == "G":
        return 2
    else:
        return 3

def num2dna(num): # DNA配列を数値に変換。nt2intみたいな。1文字ずつ
    if num == 0:
        return "A"
    elif num == 1:
        return "T"
    elif num == 2:
        return "G"
    else:
        return "C"
    
def dna2array(DNAstring):
    numarr = [] # リスト作成
    length = len(DNAstring)
    for i in range(0, length): # DNA配列長に対して、順番に1残基ずつarray化していく
        num = dna2num(DNAstring[i:i+1]) # i番目のDNA配列を数値化したもの
        if num >= 0:
            numarr.append(num) # リストに1文字ずつ追加。1つのDNA配列を数字に分けて1つのリストに入れ込む
    return numarr

def array2dna(numarr):
    DNAstring = []
    length = numarr.shape[0]
    for i in range(0, length): # DNA配列長に対して、順番に1残基ずつarray化していく
        dna = num2dna(numarr[i].argmax()) # i番目のDNA配列を数値化したもの
        DNAstring.append(dna) # リストに1文字ずつ追加。1つのDNA配列を数字に分けて1つのリストに入れ込む
    DNAstring = ''.join(DNAstring)
    return DNAstring


# In[ ]:


#ここから本番
model = load_model(trainmodel,compile=False)

for line in fas1:
    line=line.rstrip()
    length = len(line)
    n = 0
    sub = []
    #各シークエンス断片化用の格納ファイル
    while length >= int(bin) + (n*int(walk)):
        sub.append(line[n*int(walk):int(bin)+ n*int(walk)])
        n += 1
	#この時点でfragment化したシークエンス（for構文中のline）がsubに書き込まれている
    X = []
    for line2 in sub:
        OneHotArr = np.array([np.eye(4)[dna2array(line2)]])
        X.extend(OneHotArr)
    X = np.array(X)
    X = np.reshape(X,(-1, int(bin), 4, 1))#shapeはbin長と一致。
        
    predictions = model.predict(X)
    orig_names = []
    for img in X:
        orig_names.append(array2dna(img))
    
    for i in range(len(orig_names)):
        pred = predictions[i]
        outfile.write(format(pred[1],".3f") + "\t")#小数3桁で表示
    outfile.write("\n")
    


# In[ ]:


# outfile.close()
fas.close()
outfile.close()
os.system("paste prepped-OTU.txt " + out + " > wOTU-" + out)
os.system("rm prepped-OTU.txt")
os.system("rm prepped-seq.fa")
os.system("rm " + out)


# In[ ]:




