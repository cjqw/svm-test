#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ovo import *
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
    result = sequence(MAX_CLASS,(lambda x: []))
    for dataItem in dataSet:
        y = dataItem[0]
        if y < MAX_CLASS:
            result[y].append(dataItem[1])
    return result

# main procedure

data = classification(mapv(parseData,readFile(INPUT_FILE)))
test = mapv(parseData,readFile(TEST_FILE))
test = filterv(lambda x: x[0] < MAX_CLASS,test)
test = [mapv(lambda x:x[1],test), mapv(lambda x:x[0],test)]
TestOneVersusOne(data,test)
