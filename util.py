# Normal util functions
def readFile(input_file):
    """Read file via readlines"""
    with open(input_file,"r") as fin:
        data = fin.readlines()
        return data

def mapv(f,s):
    """Return a list of map result."""
    return list(map(f,s))

# DSL

def chooseData(v,dataSet,sign):
    """Choose a subset of data and mark them with sign."""
    result = []
    for value in v: result = result + dataSet[value]
    return mapv(lambda x: [x,sign],result)

def partitionData(vl,vr,dataSet):
    "Partition data by values"
    data = chooseData(vl,dataSet,1)+chooseData(vr,dataSet,-1)
    return mapv(lambda x:x[0],data), mapv(lambda x:x[1],data)
