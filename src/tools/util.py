# Normal util functions
from svmutil import *
from tools.grid import *

def identity(x): return x

def constant(c): return (lambda x: c)

def readFile(input_file):
    """Read file via readlines"""
    with open(input_file,"r") as fin:
        data = fin.readlines()
        return data

def filterv(f,s):
    """Return a list of filter result."""
    return list(filter(f,s))

def reducev(f,s):
    """Return a list of reduce result."""
    return list(reduce(f,s))

def mapv(f,s):
    """Return a list of map result."""
    return list(map(f,s))

def sequence(l,f = (lambda x: 0)):
    """Return a vector of length l,
    The vector will be initialized with f(index) (default with 0)"""
    return [f(i) for i in range(0,l)]

def matrix(row,col,f=(lambda x:0)):
    """Return a 2 dim vector which contains row vectors of col elements.
    The matrix will be initialized with f([row,col]) (default with 0)"""
    return [[f([i,j]) for j in range(0,col)] for i in range(0,row)]

def plainMatrix(row,col,f=(lambda x:x)):
    """Return a plain matrix which contains row * col elements,
    The matrix will be initialized with f([row,col]) (default with [row,col])"""
    return [f([i,j]) for i in range(0,row) for j in range(0,col)]

# DSL for homework

def chooseData(v,dataSet,sign):
    """Choose a subset of data and mark them with sign."""
    result = []
    for value in v: result = result + dataSet[value]
    return mapv(lambda x: [x,sign],result)

def partitionData(vl,vr,dataSet):
    "Partition data by values"
    data = chooseData(vl,dataSet,1)+chooseData(vr,dataSet,-1)
    return mapv(lambda x:x[0],data), mapv(lambda x:x[1],data)

def accuracy(result,answer):
    """Return a triple hit,total,acc.
    hit is the number of correct prediction,
    total is the length of input,
    acc=(hit/total)*100 is the accuracy of prediction."""
    hit = 0
    total = len(result)
    for i in range(0,total):
        if result[i] == answer[i]: hit = hit + 1
    return hit,total,(hit/total)*100

def printData(fout,x,y):
    for index in range(0,len(x)):
        item = x[index]
        label = y[index]
        output = str(label)
        for i in range(0,len(item)):
            output = output + ' ' + str(i+1)+':'+str(item[i])
        fout.write(output+'\n')

def trainModel(vl,vr,dataSet,model_name,cmd):
    x,y = partitionData(vl,vr,dataSet)
    with open('tmp','w')as fout:
        printData(fout,x,y)
    rate, param = find_parameters('tmp', '-log2c 5,16,2 -log2g 10,-3,-2 -gnuplot null -v')
    # param = {'c' : 10000, 'g': 1}
    cmd = '-c ' + str(param["c"]) + ' -g ' + str(param["g"]) + cmd
    cmd = cmd + ' -w1 ' + str(len(vr))
    cmd = cmd + ' -w-1 ' + str(len(vl))
    model = svm_train(y,x,cmd)
    svm_save_model(model_name,model)
    return model

def getModel(vl,vr,dataSet,cmd):
    """Train a model which can tell vl and vr.
    data of vl set will be marked with 1, while vr with -1.'"""
    model_name = 'models/' + str(vl) + ':' + str(vr) + cmd + ".model"
    model = svm_load_model(model_name)
    if model == None:
        model = trainModel(vl,vr,dataSet,model_name,cmd)
    return model

def testResult(m,testSet):
    "Test a model and return the accuracy."
    return svm_predict(testSet[1],testSet[0],m)[0]
