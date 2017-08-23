#
from numpy import *
import operator  # use its itemgetter() func

def createDataset():
    group = array([[1.0, 1.1],[1.0, 1.0],[0, 0],[0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

# input data vector, input dataset, labels of input dataset,  
def classify0(inpX, dataset, label, k):
    datasetSize = dataset.shape[0]
    # construct an array by repeating inpx the number of times given by arg2 tuple (datasetSize x 1)
    # pos-wise subtraction of inpX and each dataset
    diffMat = tile(inpX, (datasetSize, 1)) - dataset
    diffMatSq = diffMat ** 2  #element-wise square
    distSq = diffMatSq.sum(axis = 1) # summation of each column
    dist = distSq ** 0.5
    # sort in ascending order (default)
    sortIdx = dist.argsort()
    ### *** see the prcessing of classCount, a good example of dealing with dictionary ***
    classCount = {} # an empty dict
    for i in range(k):
        selectLabel = label[sortIdx[i]]
        classCount[selectLabel] = classCount.get(selectLabel, 0) + 1 # default is 0
    # sort in descending order
    # get the second elems of each item (label frequency)
    # iteritems: return an iterator over the dictionary's (key,value) pairs => generator of tuples
    sortedClassCount = sorted(classCount.iteritems(), key = operator.itemgetter(1), reverse = True)
    return sortedClassCount[0][0] # sortedClassCount[0] => most frequent (label, freq) pair
