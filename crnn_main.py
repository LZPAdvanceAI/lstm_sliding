from __future__ import print_function
import argparse
import random
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import numpy as np
from warpctc_pytorch import CTCLoss
import os
import utils
import dataset

import models.crnn as crnn

parser = argparse.ArgumentParser()
parser.add_argument('--trainroot', default='data/train_lmdb', help='path to dataset')
parser.add_argument('--valroot', default='data/test_lmdb', help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--nh', type=int, default=256, help='size of the lstm hidden state')
parser.add_argument('--niter', type=int, default=2500, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.1, help='learning rate for Critic, default=0.00005')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--crnn', default='', help="path to crnn (to continue training)")
parser.add_argument('--alphabet', type=str, default='abcdefghijklmnopqrstuvwxyz')
parser.add_argument('--Diters', type=int, default=5, help='number of D iters per each G iter')
parser.add_argument('--experiment', default=None, help='Where to store samples and models')
parser.add_argument('--displayInterval', type=int, default=500, help='Interval to be displayed')
parser.add_argument('--n_test_disp', type=int, default=10, help='Number of samples to display when test')
parser.add_argument('--valInterval', type=int, default=1000, help='Interval to be displayed')
parser.add_argument('--saveInterval', type=int, default=1000, help='Interval to be displayed')
parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is sgd)')
parser.add_argument('--adadelta', action='store_true', help='Whether to use adadelta (default is sgd)')
parser.add_argument('--rmsprop', action='store_true', help='Whether to use rmsprop (default is sgd)')
parser.add_argument('--keep_ratio', action='store_true', help='whether to keep ratio for image resize')
parser.add_argument('--random_sample', action='store_true', help='whether to sample the dataset with random sampler')
opt = parser.parse_args()
print(opt)

if opt.experiment is None:
    opt.experiment = 'samples'
os.system('mkdir {0}'.format(opt.experiment))

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

train_dataset = dataset.lmdbDataset(root=opt.trainroot)
assert train_dataset
if not opt.random_sample:
    sampler = dataset.randomSequentialSampler(train_dataset, opt.batchSize)
else:
    sampler = None
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=opt.batchSize,
    shuffle=True, sampler=sampler,
    num_workers=int(opt.workers))

test_dataset = dataset.lmdbDataset(root=opt.valroot)

ngpu = int(opt.ngpu)
nh = int(opt.nh)
alphabet = opt.alphabet
nclass = len(alphabet) + 1

converter = utils.strLabelConverter(alphabet)
criterion = CTCLoss()


crnn = crnn.CRNN(nclass, nh, ngpu)
if opt.crnn != '':
    print('loading pretrained model from %s' % opt.crnn)
    crnn.load_state_dict(torch.load(opt.crnn))
print(crnn)

traject = torch.LongTensor(opt.batchSize, 60)
text = torch.IntTensor(opt.batchSize * 5)
length = torch.IntTensor(opt.batchSize)

if opt.cuda:
    crnn.cuda()
    traject = traject.cuda()
    criterion = criterion.cuda()

traject = Variable(traject)
text = Variable(text)
length = Variable(length)

# loss averager
loss_avg = utils.averager()

# setup optimizer
if opt.adam:
    optimizer = optim.Adam(crnn.parameters(), lr=opt.lr,
                           betas=(opt.beta1, 0.999))
elif opt.adadelta:
    optimizer = optim.Adadelta(crnn.parameters(), lr=opt.lr)
elif opt.rmsprop:
    optimizer = optim.RMSprop(crnn.parameters(), lr=opt.lr)
else:
    optimizer = torch.optim.SGD(crnn.parameters(), lr=opt.lr,
                                momentum=0.9,
                                weight_decay=1e-4)

def padding2tensor(trajects):
    nTrajs = len(trajects)
    maxLen = max([len(trajects[i].split(',')) for i in range(len(trajects))])
    new_trajects = torch.LongTensor(nTrajs, maxLen)
    for i in range(nTrajs):
        traject = trajects[i].split(',')
        trajLen = len(traject)
        if trajLen < maxLen:
            padding = [traject[trajLen-1]] * (maxLen - trajLen)
            traject.extend(padding)
        new_trajects[i] = torch.LongTensor([long(traject[j]) for j in range(maxLen)])
    return new_trajects

def val(net, dataset, criterion, max_iter=100):
    print('Start val')

    for p in crnn.parameters():
        p.requires_grad = False

    net.eval()
    data_loader = torch.utils.data.DataLoader(
        dataset, shuffle=False, batch_size=opt.batchSize, num_workers=int(opt.workers))
    val_iter = iter(data_loader)

    i = 0
    n_correct = 0
    loss_avg = utils.averager()

    for i in range(max_iter):
        data = val_iter.next()
        i += 1
        cpu_trajects, cpu_texts = data
        cpu_trajects = padding2tensor(cpu_trajects)
        #print(cpu_trajects)
        #print(cpu_texts)
        batch_size = cpu_trajects.size(0)
        utils.loadData(traject, cpu_trajects)
        t, l = converter.encode(cpu_texts)
        utils.loadData(text, t)
        utils.loadData(length, l)

        preds = crnn(traject)
        preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
        cost = criterion(preds, text, preds_size, length) / batch_size
        loss_avg.add(cost)

        _, preds = preds.max(2)
        preds = preds.squeeze(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
        for pred, target in zip(sim_preds, cpu_texts):
            if pred == target.lower():
                n_correct += 1

    raw_preds = converter.decode(preds.data, preds_size.data, raw=True)
    for raw_pred, pred, gt in zip(raw_preds, sim_preds, cpu_texts):
        print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))

    accuracy = n_correct / float(max_iter * opt.batchSize)
    print('Test loss: %f, accuray: %f' % (loss_avg.val(), accuracy))


#  val(crnn, test_dataset, criterion)
#  exit(0)
def trainBatch(net, criterion, optimizer):
    data = train_iter.next()
    cpu_trajects, cpu_texts = data
    cpu_trajects = padding2tensor(cpu_trajects)
    #print(cpu_trajects)
    #print(cpu_texts)
    batch_size = cpu_trajects.size(0)
    utils.loadData(traject, cpu_trajects)
    t, l = converter.encode(cpu_texts)
    utils.loadData(text, t)
    utils.loadData(length, l)

    preds = crnn(traject)
    preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
    cost = criterion(preds, text, preds_size, length) / batch_size
    crnn.zero_grad()
    cost.backward()
    optimizer.step()
    return cost


#val(crnn, test_dataset, criterion)

for epoch in range(opt.niter):
    train_iter = iter(train_loader)
    i = 0
    while i < len(train_loader):
        for p in crnn.parameters():
            p.requires_grad = True
        crnn.train()

        cost = trainBatch(crnn, criterion, optimizer)
        loss_avg.add(cost)
        i += 1

        if i % opt.displayInterval == 0:
            print('[%d/%d][%d/%d] Loss: %f' %
                  (epoch, opt.niter, i, len(train_loader), loss_avg.val()))
            loss_avg.reset()

        if i % opt.valInterval == 0:
            val(crnn, test_dataset, criterion)

        # do checkpointing
        if i % opt.saveInterval == 0:
            torch.save(
                crnn.state_dict(), '{0}/netCRNN_{1}_{2}.pth'.format(opt.experiment, epoch, i))
