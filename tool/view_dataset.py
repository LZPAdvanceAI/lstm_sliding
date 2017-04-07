import lmdb

env = lmdb.open('data/train_lmdb',
                max_readers=1,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False)

with env.begin(write=False) as txn:
    nSamples = int(txn.get('num-samples'))
    #print nSamples
    for index in range(nSamples):
        traject_key = 'traject-%09d' % (index+1)
        label_key = 'label-%09d' % (index+1)
        traject = txn.get(traject_key)
        label = txn.get(label_key)
        print label,traject
