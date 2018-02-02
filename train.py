from __future__ import unicode_literals, print_function, division

import time
import random
from util import prepareData, showPlot, variableFromSentence, timeSince, variablesFromPair, filter_embedding
from encoder import EncoderRNN
from attndecoder import AttnDecoderRNN
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.autograd as autograd

SOS_token = 0
EOS_token = 1
teacher_forcing_ratio = 0.5


def rawToVectorizedOutput(decoder_output, real_id):
    numpy_version = np.zeros(lang.n_words)
    numpy_version[decoder_output] = 1

    real_output = np.zeros(lang.n_words)
    real_output[real_id.data.cpu().numpy()[0]] = 1

    return numpy_version, real_output


def get_n_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


def train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion,
          max_length=100):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda()

    loss = 0
    print("input length ", input_length)
    print("target length ", target_length)

    for ei in range(input_length):
        e_i = input_variable[ei]
        """e_i=np.array([e_i])
        enc_input=Variable(torch.LongTensor(e_i.tolist()))
        enc_input.cuda()"""
        encoder_output, encoder_hidden = encoder(e_i, encoder_hidden)
        encoder_outputs[ei] = encoder_output[0][0]
    sos_t = np.array([SOS_token])
    decoder_input = Variable(torch.LongTensor(sos_t.tolist()))
    decoder_input = decoder_input.cuda()

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)

            loss += criterion(decoder_output, target_variable[di])

            decoder_input = target_variable[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)

            loss += criterion(decoder_output, target_variable[di])

            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]  # index of most voted word----------------
            dec_ni = np.array([ni])
            decoder_input = Variable(torch.LongTensor(dec_ni.tolist()))
            decoder_input = decoder_input.cuda()

            if ni == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0] / target_length


def evaluate(encoder, decoder, sentence, max_length=40):
    input_variable = variableFromSentence(input_lang, sentence)
    input_length = input_variable.size()[0]
    encoder_hidden = encoder.initHidden()

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda()

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei],
                                                 encoder_hidden)
        encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))  # SOS
    decoder_input = decoder_input.cuda()

    decoder_hidden = encoder_hidden

    decoded_words = []
    decoder_attentions = torch.zeros(max_length, max_length)

    for di in range(max_length):
        decoder_outpcalculateLossut, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs)
        decoder_attentions[di] = decoder_attention.data
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(output_lang.index2word[ni])

        decoder_input = Variable(torch.LongTensor([[ni]]))
        decoder_input = decoder_input.cuda()

    return decoded_words, decoder_attentions[:di + 1]


def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


def trainIters(encoder, decoder, n_iters, print_every=10, plot_every=100, learning_rate=0.01):
    start = time.time()
    print("Starting training")
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [variablesFromPair(random.choice(pairs), lang)
                      for i in range(n_iters)]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_variable = training_pair[0]
        target_variable = training_pair[1]

        loss = train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))
            evaluateRandomly(encoder, decoder, 1)

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)


lang, pairs = prepareData("filtered_dataset.jsonl")
filtered_embeddings = filter_embedding(lang, "glove.6B.100d.txt")
print(random.choice(pairs))

hidden_size = 256
encoder1 = EncoderRNN(lang.n_words, hidden_size, filtered_embeddings)
attn_decoder1 = AttnDecoderRNN(hidden_size, lang.n_words, dropout_p=0.1, embeddings=filtered_embeddings)
print("parameters ", get_n_params(encoder1) + get_n_params(attn_decoder1))

encoder1.cuda()
attn_decoder1.cuda()

trainIters(encoder1, attn_decoder1, 7, print_every=100)
