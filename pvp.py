from util import *
from setting import *

def trainModel(s,dataSet,cmd):
    l,r = s
    mid = (l + r) >> 1
    vl = [i for i in range(l,mid)]
    vr = [i for i in range(mid,r)]
    return trainData(vl,vr,dataSet,cmd)

def calcModels(dataSet,cmd):
    models = [[0,MAX_CLASS]]
    now = 0
    while now < len(models):
        l,r = models[now]
        mid = (l + r) >> 1
        if l < mid and mid < r:
            models.append([l,mid])
            models.append([mid,r])
        now = now + 1
    models = filterv(lambda x: x[1] - x[0] > 1, models)
    mapv(lambda x: x.append(trainModel(x,dataSet,cmd)),models)
    return models

def classify(models,results):
    l,r = models[0][:-1]
    now = 0
    while r - l > 1:
        while models[now][0] != l or models[now][1] != r:
            now = now + 1
        mid = (l + r) >> 1
        if results[now] > 0: r = mid
        else: l = mid
    return l

def testPartVersusPart(dataSet,testSet,cmd = '-c 10000 -g 1'):
    """Train a multi-class classification model
    with part versus part method."""
    models = calcModels(dataSet,cmd)
    result = sequence(len(testSet[0]))
    class_result = mapv(lambda x: testResult(x[2],testSet),models)

    for i in range(0,len(result)):
        result[i] = classify(models,mapv(lambda x: x[i],class_result))

    hit,total,acc = accuracy(result,testSet[1])
    print("acc: ",acc,"%")
    print("hit/total: ",hit,"/",total)
