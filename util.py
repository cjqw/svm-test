# Normal util functions
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
