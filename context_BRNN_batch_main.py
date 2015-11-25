#-*- encoding: utf8 -*-

from __future__ import absolute_import

import sys

import numpy as np
np.random.seed(1337)  # for reproducibility

import theano
theano.config.optimizer='None'
import theano.tensor as T

from keras.models import make_batches
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.layers.core import Dense, Dropout, Activation, Flatten
from context_BRNN_batch import context_BRNN
from load_data import load_data, split_data, load_wordvector, sents2mat
from util import make_shared


def main():
    print '---------------------- load data ------------------'
    print('Load wordvector...')
    wordvectorMat, vocabDict, nEmbed = load_wordvector('nearby_vectors.txt')
    wordvectorMat = np.array(wordvectorMat, dtype=theano.config.floatX)

    print('Load data...')
    X, Y = load_data({'../data/NLPNN_nearby.guanggao':1}, vocabDict)
    (trainX1, trainY1, nTrain1), (validX1, validY1, nValid1), (testX1, testY1, nTest1) = split_data(X, Y, [0.8, 0.1, 0.1])
    X, Y = load_data({'../data/NLPNN_nearby.normal':0}, vocabDict)
    (trainX0, trainY0, nTrain0), (validX0, validY0, nValid0), (testX0, testY0, nTest0) = split_data(X, Y, [0.8, 0.1, 0.1])
    nTrain = nTrain0+nTrain1; nValid = nValid0+nValid1; nTest = nTest0+nTest1
    print 'train samples: %ld | valid samples: %ld | test sample: %ld ' % (nTrain, nValid, nTest)
    trainX1.extend(trainX0); trainY1.extend(trainY0); trainY = np.array(trainY1, dtype=np.int32)
    validX1.extend(validX0); validY1.extend(validY0); validY = np.array(validY1, dtype=np.int32)
    testX1.extend(testX0);   testY1.extend(testY0);   testY  = np.array(testY1,  dtype=np.int32)

    trainX, trainX_nwords_lst = sents2mat(wordvectorMat, trainX1)
    validX, validX_nwords_lst = sents2mat(wordvectorMat, validX1)
    testX,  testX_nwords_lst  = sents2mat(wordvectorMat, testX1)

    print '---------------------- theano graph ---------------'
    learning_rate = 0.01
    nContext = 50
    nHidden  = 100
    nClasses = 2
    batch_size = 64
    
    X = T.dmatrix('X') # word * embeding
    Y = T.ivector('Y')
    clf = context_BRNN(batch_size, nEmbed, nContext, nHidden, nClasses, learning_rate=learning_rate, loss_func = 'binary_crossentropy')

    print '---------------------- train ----------------------'
    nepoches = 100
    improve_ratio = 0.995
    patience = 600
    validation_freq = 1

    train_batch_lst = make_batches(nTrain, batch_size)
    valid_batch_lst = make_batches(nValid, batch_size)
    test_batch_lst  = make_batches(nTest,  batch_size)


    best_cost = np.inf
    done = False
    epoch = 0
    iteration = 0
    while epoch <= nepoches and (not done):
          epoch += 1
          for batch, sents_idx_lst in enumerate(train_batch_lst):
              iteration += 1
              word_start, word_end, sents_start, sents_end, nwords_lst2 = batchRange(trainX_nwords_lst, sents_idx_lst)
              train_cost = clf.train(trainX[word_start: word_end, :], nwords_lst2, trainY[sents_start : sents_end])
              print 'train_cost: %g' % train_cost
              validate_cost = []
              if (iteration+1) % validation_freq == 0:
                 for idx_lst in valid_batch_lst:
                     #print 'validation...'
                     word_start, word_end, sents_start, sents_end, nwords_lst2 = batchRange(validX_nwords_lst, idx_lst)
                     validate_cost.append(clf.cost(validX[word_start: word_end, :], nwords_lst2, validY[sents_start : sents_end]))
                 validate_cost = np.mean(validate_cost)
                 print 'epoch: %5d | iter: %5d | validate_cost: %5.3g | improve_ratio: %5g' %  (epoch, iteration, validate_cost, validate_cost / best_cost)
                 sys.stdout.flush()
                 if validate_cost < best_cost * improve_ratio:
          if patience <= iteration:
             done = True
             test_error = []
             for idx_lst in test_batch_lst:
                 word_start, word_end, sents_start, sents_end, nwords_lst2 = batchRange(testX_nwords_lst, idx_lst)
                 test_error.append(clf.error(testX[word_start: word_end, :], nwords_lst2, testY[sents_start : sents_end]))
             test_error = np.sum(test_error)/float(nTest)
             print '\n\nDONE: epoch %5d | iter: %5d | best_error: %5g\n\n' % (epoch, iteration, test_error)
             break  


def batchRange(nWords_lst, idx_lst): 
    sent_start = idx_lst[0]
    sent_end = idx_lst[1]
    nwords_lst2 = np.copy(nWords_lst[sent_start: sent_end + 1]) 
    word_start = nwords_lst2[0]
    word_end   = nwords_lst2[-1]
    nwords_lst2 -= word_start
    return word_start, word_end, sent_start, sent_end, nwords_lst2

def eval_batches(fn, nWords_lst, batch_pieces_lst):
    res = []
    for idx_lst in valid_batch_lst:
        print idx_lst
        word_start, word_end, sents_start, sents_end, nwords_lst = batchRange(nWords_lst, idx_lst)
        print word_start, word_end, sents_start, sents_end
        res.append(fn(sents_start, sents_end, word_start, word_end, nwords_lst))
    res = np.mean(validate_cost)


if __name__ == '__main__':
   main()
