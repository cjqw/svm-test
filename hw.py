from svmutil import *
from util import *
from setting import *

def trainData(vl,vr,dataSet,cmd = '-c 10 -g 1'):
    """Train a model which can tell vl and vr.
    Default command is '-c 10 -g 1'"""
    x,y = partitionData(vl,vr,dataSet)
    return svm_train(y,x,cmd)

def TestOneVersusOne(dataSet,TestSet,cmd = '-c 10 -g 1'):
    """Train a multi-class classification model
    with one versus one method."""
    models = [[i,j] for i in range(0,MAX_CLASS) for j in range(0,MAX_CLASS)]
    mapv(lambda x: x.append(trainData(x[:1],x[1:],dataSet,cmd)),models)
