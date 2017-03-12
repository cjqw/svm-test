from util import *
from setting import *

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
    hit,total,acc = accuracy(result,testSet[1])
    print("acc: ",acc,"%")
    print("hit/total: ",hit,"/",total)
