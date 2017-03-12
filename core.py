#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from svmutil import *
from hw import *
from random import shuffle
from util import *
from setting import *

# parse the data into wanted format
def parseDims(s):
    return mapv(lambda x: float(x[x.find(":")+1:]), s.split(" "))

def parseData(s):
    value = int(s[:s.find(" ")])
    dims = parseDims(s[s.find(" ") + 1:-1])
    return [value,dims]

def classification(dataSet):
    result = [[] for i in range(0,MAX_CLASS)]
    for dataItem in dataSet:
        y = dataItem[0]
        if y < MAX_CLASS:
            result[y].append(dataItem[1])
    return result

def testModel(vl,vr,m):
    "Test a model and return the accuracy."
    x,y = partitionData(vl,vr,test)
    result = svm_predict(y,x,m)[1]
    return result[0]

# main procedure

data = classification(mapv(parseData,readFile(INPUT_FILE)))
test = classification(mapv(parseData,readFile(TEST_FILE)))
TestOneVersusOne(data,test)
