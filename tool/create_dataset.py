import os
import lmdb # install lmdb by "pip install lmdb"
import numpy as np

def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.iteritems():
            txn.put(k, v)


#def createDataset(outputPath, imagePathList, labelList, lexiconList=None, checkValid=True):
def createDataset(outputPath, labelPath):
    """
    Create LMDB dataset for Sliding LSTM training.

    ARGS:
        outputPath    : LMDB output path
        labelPath     : groundtruth texts
    """
    with open(labelPath, 'r') as file:
        labelList = file.readlines()

    nSamples = len(labelList)
    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 1
    for i in xrange(nSamples):
        label, traject = labelList[i].split()
        trajectKey = 'traject-%09d' % cnt
        labelKey = 'label-%09d' % cnt
        cache[trajectKey] = traject
        cache[labelKey] = label
        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt-1
    cache['num-samples'] = str(nSamples)
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)


if __name__ == '__main__':
    #createDataset('data/train_lmdb', 'train.txt')
    createDataset('data/test_lmdb', 'test.txt')
