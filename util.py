
import unicodedata
import re
from Lang import Lang
from load_glove import get_embeddings
from torch.autograd import Variable
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
MAX_LENGTH = 100

EOS_token = 1

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def stringToInput(string):
    string = normalizeString(string)

    return string


def readLangs(training_set):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open('%s' % (training_set), encoding='utf-8'). \
        read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]


    lang = Lang('english')

    return lang, pairs


def filterPair(p):
    if len(p)==1:
        print(p)
    return len(p[0].split(' ')) < MAX_LENGTH and \
           len(p[1].split(' ')) < MAX_LENGTH


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


def prepareData(training_set):
    lang, pairs = readLangs(training_set)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        lang.addSentence(pair[0])
        lang.addSentence(pair[1])
    print("Counted words:")
    print(lang.name, lang.n_words)
    return lang, pairs


def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def variableFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    result = Variable(torch.LongTensor(indexes).view(-1, 1))
    return result.cuda()


def variablesFromPair(pair,lang):
    input_variable = variableFromSentence(lang, pair[0])
    target_variable = variableFromSentence(lang, pair[1])
    return (input_variable, target_variable)


import time
import math


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


def filter_embedding(lang,filename):
    glove_embeddings=get_embeddings(filename)#dict
    embed_len=len(glove_embeddings['the'])
    embeddings=np.zeros((lang.n_words,embed_len))

    new_words=0 #257
    for i in range(lang.n_words):
        word=lang.getWordFromId(i)
        if word in glove_embeddings:
            embeddings[i]=glove_embeddings[word]
        else:
            embeddings[i]=np.random.rand(embed_len)
            new_words+=1
            print(word)
    print("new words: ",new_words)
    return torch.from_numpy(embeddings)
