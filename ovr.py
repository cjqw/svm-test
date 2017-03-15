from tools.util import *
from setting import *

def rest(x):
    return filterv(lambda y: x != y,sequence(MAX_CLASS,lambda x:x))

def fill(label,new,old):
    if old >= 0: return old
    if new > 0: return label
    return -1

def testAccuracy(item,dataSet):
    num,model = item
    xx,yy = partitionData([num],rest(num),dataSet)
    a,b,c = svm_predict(yy,xx,model)
    return [b[0],num,model]

def testOneVersusRest(dataSet,testSet,cmd = ''):
    """Train a multi-class classification model
    with one versus rest method."""
    models = mapv(lambda x: [x,getModel([x],rest(x),dataSet,cmd)],sequence(MAX_CLASS,identity))
    models = mapv(lambda x: testAccuracy(x,dataSet),models)
    models = sorted(models)
    result = sequence(len(testSet[0]),lambda x: -1)
    for i in range(0,MAX_CLASS):
        index = MAX_CLASS - i - 1
        model = models[index][2]
        v = testResult(model,testSet)
        result = list(map(fill,sequence(len(result),constant(models[index][1])),v,result))
    hit,total,acc = accuracy(result,testSet[1])
    print("acc: ",acc,"%")
    print("hit/total: ",hit,"/",total)
