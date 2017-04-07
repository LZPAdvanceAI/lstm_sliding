import torch.nn as nn
import torch.nn.parallel


class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut, ngpu):
        super(BidirectionalLSTM, self).__init__()
        self.ngpu = ngpu

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        #gpu_ids = None
        gpu_ids = [0]
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            gpu_ids = range(self.ngpu)
        recurrent, _ = nn.parallel.data_parallel(
            self.rnn, input, gpu_ids)  # [T, b, h * 2]

        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)
        output = nn.parallel.data_parallel(
            self.embedding, t_rec, gpu_ids)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output


class CRNN(nn.Module):

    def __init__(self, nclass, nh, ngpu):
        super(CRNN, self).__init__()
        self.ngpu = ngpu

        cnn = nn.Sequential()
        cnn.add_module('embedding', nn.Embedding(26,256))

        self.cnn = cnn
        self.rnn = nn.Sequential(
            BidirectionalLSTM(256, nh, nh, ngpu),
            BidirectionalLSTM(nh, nh, nclass, ngpu)
        )

    def forward(self, input):
        #gpu_ids = None
        gpu_ids = [0]
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            gpu_ids = range(self.ngpu)

        # conv features
        conv = nn.parallel.data_parallel(self.cnn, input, gpu_ids)
        conv = conv.permute(1, 0, 2)  # [w, b, c]

        # rnn features
        output = nn.parallel.data_parallel(self.rnn, conv, gpu_ids)

        return output
