from util import *
from setting import *

def rest(x):
    return filterv(lambda y: x != y,sequence(MAX_CLASS,lambda x:x))

def fill(label,new,old):
    if old >= 0: return old
    if new > 0: return label
    return -1

def TestOneVersusRest(dataSet,testSet,cmd = '-c 10 -g 1'):
    """Train a multi-class classification model
    with one versus rest method."""
    models = mapv(lambda x: trainData([x],rest(x),dataSet,cmd),sequence(MAX_CLASS,lambda x: x))
    result = sequence(len(testSet[0]),lambda x: -1)
    for i in range(0,MAX_CLASS):
        model = models[i]
        v = testResult(model,testSet)
        result = list(map(fill,sequence(len(result),lambda x: i),v,result))
    hit,total,acc = accuracy(result,testSet[1])
    print("acc: ",acc,"%")
    print("hit/total: ",hit,"/",total)
