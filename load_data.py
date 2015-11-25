#-*- encoding: utf8 -*-
__author__ = 'root'

import numpy as np
import theano
seed = 20151116

def load_data(txt_lable_dict, vocab_dict):
    X = []
    Y = []
    for txt, label in txt_lable_dict.iteritems():
        with open(txt, 'rb') as fid:
            for line in fid:
                tmp = line.strip().split()
                x = []
                if tmp:
                    for w in tmp:
                        idx = vocab_dict.get(w, None)
                        if idx:
                            x.append(idx)
                    if x and len(x) > 0:
                        X.append(x)
                        Y.append(label)
    assert len(X)==len(Y), 'len(X)!=len(Y)'
    return X, Y

def split_data(X, Y, ratio):
    assert len(np.unique(Y)) == 1, 'more than two labels'
    nsents = len(X)
    ntrain = int(nsents * ratio[0])
    nvalid = int(nsents * ratio[1])
    ntest  = nsents - ntrain - nvalid
    trainX = X[:ntrain]; trainY = Y[:ntrain]
    validX = X[ntrain:(ntrain+nvalid)]; validY = Y[ntrain:(ntrain+nvalid)]
    testX  = X[-ntest:]; testY = Y[:ntest:]
    return (trainX, trainY, ntrain), (validX, validY, nvalid), (testX, testY, ntest)

def load_wordvector(txt):
    wordvectorMat = []
    vocabDict = dict()
    lineno = -1
    with open(txt, 'rb') as fid:
        for line in fid:
            tmp = line.strip().split()
            if tmp:
               lineno += 1
               if lineno == 0:
                  embed_dim = int(tmp[1])
                  continue
               word = tmp[0]
               vocabDict[word] = lineno - 1
               wordvectorMat.append([float(x) for x in tmp[1:]])
    return wordvectorMat, vocabDict, embed_dim

def sents2mat(wordvectorMat, sents_lst):
    res = []
    nWords_lst = [0]
    cumsum = 0
    for sent in sents_lst:
         for w_idx in sent:
             res.append(wordvectorMat[w_idx,:])
         nWords_lst.append(len(res))
    return np.array(res, dtype=theano.config.floatX), np.array(nWords_lst, dtype=np.int32)


