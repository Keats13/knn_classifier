# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 22:01:33 2017

@author: lenovo
"""
import numpy as np
import operator

#（1）knn
#clf = skl.neighbors.KNeighborsClassifier(algorithm='kd_tree')  
#clf.fit(traindataset, localdataset)  
def classify_knn(inX, dataset, labels, k):  
    """ 
    inX 是输入的测试样本，是一个[x, y]样式的 
    dataset 是训练样本集 
    labels 是训练样本标签 
    k 是top k最相近的 
    """  
    # shape返回矩阵的[行数，列数]，  
    # 那么shape[0]获取数据集的行数，  
    # 行数就是样本的数量  
    dataSetSize = dataset.shape[0]   
      
    """ 
    下面的求距离过程就是按照欧氏距离的公式计算的。 
    即 根号(x^2+y^2) 
    """  
    # tile属于numpy模块下边的函数  
    # tile（A, reps）返回一个shape=reps的矩阵，矩阵的每个元素是A  
    # 比如 A=[0,1,2] 那么，tile(A, 2)= [0, 1, 2, 0, 1, 2]  
    # tile(A,(2,2)) = [[0, 1, 2, 0, 1, 2],  
    #                  [0, 1, 2, 0, 1, 2]]  
    # tile(A,(2,1,2)) = [[[0, 1, 2, 0, 1, 2]],  
    #                    [[0, 1, 2, 0, 1, 2]]]   
    # 上边那个结果的分开理解就是：  
    # 最外层是2个元素，即最外边的[]中包含2个元素，类似于[C,D],而此处的C=D，因为是复制出来的  
    # 然后C包含1个元素，即C=[E],同理D=[E]  
    # 最后E包含2个元素，即E=[F,G],此处F=G，因为是复制出来的  
    # F就是A了，基础元素  
    # 综合起来就是(2,1,2)= [C, C] = [[E], [E]] = [[[F, F]], [[F, F]]] = [[[A, A]], [[A, A]]]  
    # 这个地方就是为了把输入的测试样本扩展为和dataset的shape一样，然后就可以直接做矩阵减法了。  
    # 比如，dataset有4个样本，就是4*2的矩阵，输入测试样本肯定是一个了，就是1*2，为了计算输入样本与训练样本的距离  
    # 那么，需要对这个数据进行作差。这是一次比较，因为训练样本有n个，那么就要进行n次比较；  
    # 为了方便计算，把输入样本复制n次，然后直接与训练样本作矩阵差运算，就可以一次性比较了n个样本。  
    # 比如inX = [0,1],dataset就用函数返回的结果，那么  
    # tile(inX, (4,1))= [[ 0.0, 1.0],  
    #                    [ 0.0, 1.0],  
    #                    [ 0.0, 1.0],  
    #                    [ 0.0, 1.0]]  
    # 作差之后  
    # diffMat = [[-1.0,-0.1],  
    #            [-1.0, 0.0],  
    #            [ 0.0, 1.0],  
    #            [ 0.0, 0.9]]  
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataset  
      
    # diffMat就是输入样本与每个训练样本的差值，然后对其每个x和y的差值进行平方运算。  
    # diffMat是一个矩阵，矩阵**2表示对矩阵中的每个元素进行**2操作，即平方。  
    # sqDiffMat = [[1.0, 0.01],  
    #              [1.0, 0.0 ],  
    #              [0.0, 1.0 ],  
    #              [0.0, 0.81]]  
    sqDiffMat = diffMat ** 2  
      
    # axis=1表示按照横轴，sum表示累加，即按照行进行累加。  
    # sqDistance = [[1.01],  
    #               [1.0 ],  
    #               [1.0 ],  
    #               [0.81]]  
    sqDistance = sqDiffMat.sum(axis=1)  
      
    # 对平方和进行开根号  
    distance = sqDistance ** 0.5  
      
    # 按照升序进行快速排序，返回的是原数组的下标。  
    # 比如，x = [30, 10, 20, 40]  
    # 升序排序后应该是[10,20,30,40],他们的原下标是[1,2,0,3]  
    # 那么，numpy.argsort(x) = [1, 2, 0, 3]  
    sortedDistIndicies = distance.argsort()  
      
    # 存放最终的分类结果及相应的结果投票数  
    classCount = {}  
      
    # 投票过程，就是统计前k个最近的样本所属类别包含的样本个数  
    for i in range(k):  
        # index = sortedDistIndicies[i]是第i个最相近的样本下标  
        # voteIlabel = labels[index]是样本index对应的分类结果('A' or 'B')  
        voteIlabel = labels[sortedDistIndicies[i]]  
        # classCount.get(voteIlabel, 0)返回voteIlabel的值，如果不存在，则返回0  
        # 然后将票数增1  
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1  
      
    # 把分类结果进行排序，然后返回得票数最多的分类结果  
    # classCount.iteritems()生成迭代器，形成这样的二元组list：［(key,value),(key,value),...］
    # sorted(iterable, cmp=None, key=None, reverse=False)
    # iterable：指定为要排序的list或iterable对象
    # cmp：接受一个函数(有两个参数)，指定排序时进行比较的函数，可以指定一个函数或lambda表达式，
    # key：接受一个函数(只有一个参数)，指定待排序元素的哪一项进行排序
    # 1.operator.itemgetter(1)指定按照第二个域进行排序；2.key=lambda student:student[2]
    # reverse为False是升序排列，为True是降序排列
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)  
    return sortedClassCount[0][0]  