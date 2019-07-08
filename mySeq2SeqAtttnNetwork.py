#!/usr/bin/python

from __future__ import unicode_literals, print_function, division
import os
import mySeq2SeqPreprocessing as pp
import mySeq2SeqEvaluating as eval
import random
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import time
import math
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1

MAX_LENGTH = 300

class EncoderRNN(nn.Module):
    """
    Encoder is a GRU that outputs some value for every word in the input sentence. For each input word, it outputs a vector and a hidden state,
    and the hidden state is used for the next input word.
    """
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(-1, 1, hidden_size) #input into embedding
        output = embedded
        output, hidden = self.gru(output, hidden) #gru takes embedded and previous hidden to make new output and hidden
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(-1, 1, hidden_size) #input goes into embedding, then into dropout because it has already been processed in the encoder
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1) #softmax onto concat'd tensor using decoder input and hidden state to make attn weights
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), # bmm (dot product) from attn weights and encoder outputs
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1) #put embedded input and the outcome of the attention applied encoder outputs for the attn combined output
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output) #relu on this value
        output, hidden = self.gru(output, hidden) #gru on output and previous hidden to make new output and hidden value

        output = F.log_softmax(self.out(output[0]), dim=1) #log softmax on linear layer of output
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


teacher_forcing_ratio = 0.5

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad() #zero the gradients
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length): # for index in input length
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length



def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.savefig("../myfig")


def trainIters(encoder, decoder, epoch, print_every=100, plot_every=10, learning_rate=0.01, batch_size=16):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    myDataset = pp.SentenceDataset(pairs)
    train_loader = DataLoader(myDataset, batch_size)
    criterion = nn.NLLLoss()

    for batch_idx, (data, target) in enumerate(train_loader):
        for i in range(batch_size):
            loss = train(pp.tensorFromSentence(input_lang, data[i]), pp.tensorFromSentence(output_lang, target[i]), encoder,
                         decoder, encoder_optimizer, decoder_optimizer, criterion)
            print_loss_total += loss
            plot_loss_total += loss

        if batch_idx % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print(batch_idx, print_loss_avg)

        if batch_idx % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    hidden_size = 256
    epoch_count = 5
    input_lang, output_lang, pairs = pp.prepareData('eng', 'chinese', True)
    print(random.choice(pairs))
    encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)
    for j in range(epoch_count):
        trainIters(encoder1, attn_decoder1, j, print_every=100, plot_every=10)
    torch.save(encoder1.state_dict(), "../myencoder1")
    torch.save(attn_decoder1.state_dict(), "../mydecoder1")
    eval.evaluateRandomly(input_lang, output_lang, encoder1, attn_decoder1, pairs)