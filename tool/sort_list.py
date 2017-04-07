with open('train.txt', 'r') as file:
    labelList = file.readlines()

labelList = sorted(labelList, key=lambda line: len(line.split(',')))
f = open('train_sort.txt', 'w')
nSamples = len(labelList)
for i in xrange(nSamples):
    print >> f, labelList[i]
f.close()
