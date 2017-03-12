from svmutil import *
from util import *
from setting import *

def trainData(vl,vr,dataSet,cmd = '-c 10 -g 1'):
    """Train a model which can tell vl and vr.
    data of vl set will be marked with 1, while vr with -1
    Default command is '-c 10 -g 1'"""
    x,y = partitionData(vl,vr,dataSet)
    return svm_train(y,x,cmd)

def testResult(m,testSet):
    "Test a model and return the accuracy."
    return svm_predict(testSet[1],testSet[0],m)[0]

def vote(x,l,r):
    if x > 0:return l
    else:return r

def combine(x,y):
    x.append(y)
    return x

def maxOccurence(s):
    c = sequence(MAX_CLASS)
    for predict in s: c[predict] = c[predict] + 1
    for i in range(MAX_CLASS):
        if c[i] == max(c): return i

def TestOneVersusOne(dataSet,testSet,cmd = '-c 10 -g 1'):
    """Train a multi-class classification model
    with one versus one method."""
    models = filterv(lambda x: x[0] < x[1] , plainMatrix(MAX_CLASS,MAX_CLASS))
    mapv(lambda x: x.append(trainData(x[:1],x[1:],dataSet,cmd)),models)
    result = sequence(len(testSet[0]),lambda x: [])
    for model in models:
        l,r,m = model
        v = mapv(lambda x: vote(x,l,r),testResult(m,testSet))
        result = list(map(combine,result,v))
    result = mapv(maxOccurence,result)
    hit = 0
    total = len(result)
    for i in range(0,total):
        if result[i] == testSet[1][i]:
            hit = hit + 1
    print(hit/total)
    print("hit/total: ",hit,"/",total)
