import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import sys
sys.path.append('vqa/external/skip-thoughts.torch/pytorch')
import skipthoughts


def process_lengths(input):
    max_length = input.size(1)
    lengths = list(max_length - input.data.eq(0).sum(1).squeeze())
    return lengths

def select_last(x, lengths):
    batch_size = x.size(0)
    seq_length = x.size(1)
    mask = x.data.new().resize_as_(x.data).fill_(0)
    for i in range(batch_size):
        mask[i][lengths[i]-1].fill_(1)
    mask = Variable(mask)
    x = x.mul(mask)
    x = x.sum(1).view(batch_size, x.size(2))
    return x

class LSTM(nn.Module):

    def __init__(self, vocab, emb_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        self.vocab = vocab
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(num_embeddings=len(self.vocab)+1,
                                      embedding_dim=emb_size,
                                      padding_idx=0)
        self.rnn = nn.LSTM(input_size=emb_size, hidden_size=hidden_size, num_layers=num_layers)

    def forward(self, input):
        lengths = process_lengths(input)
        x = self.embedding(input) # seq2seq
        output, hn = self.rnn(x)
        output = select_last(output, lengths)
        return output


class TwoLSTM(nn.Module):

    def __init__(self, vocab, emb_size, hidden_size):
        super(TwoLSTM, self).__init__()
        self.vocab = vocab
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(num_embeddings=len(self.vocab)+1,
                                      embedding_dim=emb_size,
                                      padding_idx=0)
        self.rnn_0 = nn.LSTM(input_size=emb_size, hidden_size=hidden_size, num_layers=1)
        self.rnn_1 = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=1)

    def forward(self, input):
        lengths = process_lengths(input)
        x = self.embedding(input) # seq2seq
        x = getattr(F, 'tanh')(x)
        x_0, hn = self.rnn_0(x)
        vec_0 = select_last(x_0, lengths)

        # x_1 = F.dropout(x_0, p=0.3, training=self.training)
        # print(x_1.size())
        x_1, hn = self.rnn_1(x_0)
        vec_1 = select_last(x_1, lengths)
        
        vec_0 = F.dropout(vec_0, p=0.3, training=self.training)
        vec_1 = F.dropout(vec_1, p=0.3, training=self.training)
        output = torch.cat((vec_0, vec_1), 1)
        return output
        

def factory(vocab_words, opt):
    if opt['arch'] == 'skipthoughts':
        st_class = getattr(skipthoughts, opt['type'])
        seq2vec = st_class(opt['dir_st'],
                           vocab_words,
                           dropout=opt['dropout'],
                           fixed_emb=opt['fixed_emb'])
    elif opt['arch'] == '2-lstm':
        seq2vec = TwoLSTM(vocab_words,
                          opt['emb_size'],
                          opt['hidden_size'])
    elif opt['arch'] == 'lstm':
        seq2vec = TwoLSTM(vocab_words,
                          opt['emb_size'],
                          opt['hidden_size'],
                          opt['num_layers'])
    else:
        raise NotImplementedError
    return seq2vec


if __name__ == '__main__':

    vocab = ['robots', 'are', 'very', 'cool', '<eos>', 'BiDiBu']
    lstm = TwoLSTM(vocab, 300, 1024)

    input = Variable(torch.LongTensor([
        [1,2,3,4,5,0,0],
        [6,1,2,3,3,4,5],
        [6,1,2,3,3,4,5]
    ]))
    output = lstm(input)
