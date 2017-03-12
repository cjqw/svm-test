# util functions

def readFile(input_file):
    with open(input_file,"r") as fin:
        data = fin.readlines()
        return data

def mapv(f,s):
    return list(map(f,s))
