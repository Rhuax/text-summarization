
from __future__ import print_function
from __future__ import division
from __future__ import print_function
import numpy as np

import torch
def get_vocabs(dict_file):
    vocab = dict()
    line_num = 1
    for l in open(dict_file).readlines():
        l_split = l.decode('utf8').rstrip().split()
        if len(l_split) != 2:
            print ("Line %s error" % line_num)
        else:
            if l_split[0] in vocab:
                print(l_split[0].encode('utf8'), ' duplicate word.')
            vocab[l_split[0]] = int(l_split[1])
        line_num += 1
    print("Got {} words from {}".format(len(vocab), dict_file))

    return vocab

def get_embeddings(file):
    embs = dict()
    i=0
    print("Loading GloVe...")
    for l in open(file).readlines():
        i+=1
        if i%100000==0:
            print(i)
        l_split = l.strip().split()
        if len(l_split) == 2:
            continue
        embs[l_split[0]] = [float(em) for em in l_split[1:]]
    print("Got {} embeddings from {}".format(len(embs), file))

    return embs

def match_embeddings(vocab, emb):
    filtered_embeddings = np.zeros((len(vocab), len(emb.itervalues().next())))
    for w, w_id in vocab.iteritems():
        if w in emb:
            filtered_embeddings[w_id] = emb[w]
        else:
            print(u"{} not found".format(w))

    return torch.Tensor(filtered_embeddings)



