# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 16:14:36 2017

@author: lenovo
"""

import scipy.io as sio  
import numpy as np  
import func_knn as fknn
import os
# 模型：一共有56个格点，每个格点上训练数据100组，测试数据10组
# 读取训练测试数据组，  
path = os.path.abspath('.')# 获取当前路径
pathtrain=os.path.join(path,'train.mat' )# os.path.join路径拼接 
traindataset=sio.loadmat(pathtrain).get('data')#训练数据
pathtest=os.path.join(path,'test.mat' ) 
testdataset=sio.loadmat(pathtest).get('X_test')#测试数据
pathlocal=os.path.join(path,'dist1.mat' ) 
local=sio.loadmat(pathlocal).get('dist')#56个格点二维坐标

grid_num = 56 # 室内环境划分格点
grid_train_sample = 100 # 每个格点上用于训练的样本数
grid_test_sample = 10 # 每个格点上用于测试的样本数


def get_trainlabelset():
    """
    获取训练数据对应的真实格点序号
    数据排列顺序为：
        第grid_train_sample个为格点1数据，
        第grid_train_sample+1:2*grid_train_sample为格点2数据...依次类推
    """
    trainlabelset1=[]
    for grid in range(grid_num):
        for sample in range(grid_train_sample):
    #        local_tmp=local[grid]
            local_tmp=grid+1
            trainlabelset1.append(local_tmp)
    trainlabelset = np.array(trainlabelset1)
    return trainlabelset
    

def get_testlabelset():
    """
    获取测试数据对应的真实格点序号
    """
    testlabelset1=[]
    for i in range(grid_num):
        for j in range(grid_test_sample):
            testlabelset1.append(i+1)
    testlabelset = np.array(testlabelset1)
    return testlabelset
    
def createDataSet():  
    """ 
    函数作用：构建一组训练数据（训练样本）
    同时给出了这些样本的标签，及labels 
    """  
    group = traindataset
    labels = get_trainlabelset()
    return group, labels  
        
def func_rmse(predict_label,real):
    """
    获取均方根误差值
    rmse = sqrt(∑(||真实值-估计值||^2)/n)
    """
    tmp = predict_label-np.ones(np.shape(predict_label))
    predict_2d = local[map(int,tmp)]
    tmp_real = real-np.ones(np.shape(predict_label))
    real_2d = local[map(int,tmp_real)]
    norm_sqrt = (np.linalg.norm(predict_2d-real_2d,axis=-1))**2
    rmse = np.sqrt(sum(norm_sqrt)/np.shape(predict_label))
    return rmse
    
    
if __name__== "__main__":  
    # 导入数据  
    dataset, labels = createDataSet() 
    # 使用knn对测试数据集进行预测
    predict_knn=[]
    for i in range(np.shape(testdataset)[0]):
        inX = testdataset[i,:]
        # 简单分类  
        className = fknn.classify_knn(inX, dataset, labels, 100)  
        predict_knn.append(className)
    testlabelset = get_testlabelset()
    rmse = func_rmse(predict_knn,testlabelset)# 获取测试数据集上的均方根误差
