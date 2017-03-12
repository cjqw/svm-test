#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from svmutil import *
from random import shuffle
from util import *

input_file = "data/train.txt"
test_file = "data/test.txt"
max_class = 12

def parseDims(s):
    return mapv(lambda x: float(x[x.find(":")+1:]), s.split(" "))

def parseData(s):
    value = int(s[:s.find(" ")])
    dims = parseDims(s[s.find(" ") + 1:-1])
    return [value,dims]

def classification(dataSet):
    result = [[] for i in range(0,max_class)]
    for dataItem in dataSet:
        y = dataItem[0]
        result[y].append(dataItem[1])
    return result

def chooseData(v,dataSet,sign):
    result = []
    for value in v: result = result + dataSet[value]
    return mapv(lambda x: [x,sign],result)

def partitionData(vl,vr,dataSet):
    data = chooseData(vl,dataSet,1)+chooseData(vr,dataSet,-1)
    return mapv(lambda x:x[0],data), mapv(lambda x:x[1],data)

def testModel(vl,vr,m):
    x,y = partitionData(vl,vr,test)
    result = svm_predict(y,x,m)[1]
    return result[0]

data = classification(mapv(parseData,readFile(input_file)))
test = classification(mapv(parseData,readFile(test_file)))
vl = [1,2]
vr = [4,5,6]
x,y = partitionData(vl,vr,data)
m = svm_train(y,x,'-c 10')
print(testModel(vl,vr,m))

# >>> from svmutil import *
# # Read data in LIBSVM format
# >>> y, x = svm_read_problem('../heart_scale')
# >>> m = svm_train(y[:200], x[:200], '-c 4')
# >>> p_label, p_acc, p_val = svm_predict(y[200:], x[200:], m)

# # Construct problem in python format
# # Dense data
# >>> y, x = [1,-1], [[1,0,1], [-1,0,-1]]
# # Sparse data
# >>> y, x = [1,-1], [{1:1, 3:1}, {1:-1,3:-1}]
# >>> prob  = svm_problem(y, x)
# >>> param = svm_parameter('-t 0 -c 4 -b 1')
# >>> m = svm_train(prob, param)

# # Precomputed kernel data (-t 4)
# # Dense data
# >>> y, x = [1,-1], [[1, 2, -2], [2, -2, 2]]
# # Sparse data
# >>> y, x = [1,-1], [{0:1, 1:2, 2:-2}, {0:2, 1:-2, 2:2}]
# # isKernel=True must be set for precomputed kernel
# >>> prob  = svm_problem(y, x, isKernel=True)
# >>> param = svm_parameter('-t 4 -c 4 -b 1')
# >>> m = svm_train(prob, param)
# # For the format of precomputed kernel, please read LIBSVM README.


# # Other utility functions
# >>> svm_save_model('heart_scale.model', m)
# >>> m = svm_load_model('heart_scale.model')
# >>> p_label, p_acc, p_val = svm_predict(y, x, m, '-b 1')
# >>> ACC, MSE, SCC = evaluations(y, p_label)

# # Getting online help
# >>> help(svm_train)

# The low-level use directly calls C interfaces imported by svm.py. Note that
# all arguments and return values are in ctypes format. You need to handle them
# carefully.

# >>> from svm import *
# >>> prob = svm_problem([1,-1], [{1:1, 3:1}, {1:-1,3:-1}])
# >>> param = svm_parameter('-c 4')
# >>> m = libsvm.svm_train(prob, param) # m is a ctype pointer to an svm_model
# # Convert a Python-format instance to svm_nodearray, a ctypes structure
# >>> x0, max_idx = gen_svm_nodearray({1:1, 3:1})
# >>> label = libsvm.svm_predict(m, x0)
