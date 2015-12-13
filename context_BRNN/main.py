#-*- encoding: utf8 -*-
__author__ = 'root'

import cPickle as pickle

import numpy as np
import theano
#theano.config.optimizer='None' # debug
import theano.tensor as T

from keras.models import Sequential
from keras.layers.core import Dense

from context_BRNN.layers.BiContextLayer import BiContextLayer
from context_BRNN.layers.LookUpEmbedingLayer import LookUpEmbeddingLayer
from context_BRNN.layers.ElemMaxPool import ElemMaxPool
from context_BRNN.data.load_data import load_data, load_wordvector
from utils.theano_util import make_shared
from utils.padding import pad_zero
from utils.slice import binary_split
from utils.shuffle import shuffle


#------------------------------ config ----------------------------
max_seq_len = 1000
batch_size = 32
nb_epoch = 100

#------------------------------ load data -------------------------
print 'load WordVectorMat...'
WordVectorMat, vocabDict = load_wordvector('context_BRNN/data/nearby_vectors.txt')
WordVectorMat = np.array(WordVectorMat, dtype=theano.config.floatX)
print WordVectorMat.shape
# WordVectorMat = theano.shared(
#     value=WordVectorMat
# )

print 'load data...'
X, Y = load_data({
                  'context_BRNN/data/spam.100':1,
                  'context_BRNN/data/no_spam.100':0
                 },
                 vocabDict
)
print 'pad data...'
X, Y, mask_useless = pad_zero(X, Y, maxlen=max_seq_len)
print 'split data...'
(trainX, testX), (trainY, testY) = binary_split([X, Y], 0.9)
shuffle([trainX, trainY, testX, testY])
print 'train: %ld | test: %ld' % (len(trainY), len(testY))

#------------------------------ graph -----------------------------
print 'build nn...'

model = Sequential()
model.add(LookUpEmbeddingLayer(
        max_seq_len, WordVectorMat, input_length=max_seq_len, mask_zero=True
    )
)
model.add(BiContextLayer(200, 30))
model.add(ElemMaxPool(100, 1))
model.add(Dense(1))

print 'compile nn...'
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              class_mode="binary")

#----------------------- train & eval ---------------------------------
print("Train...")
model.fit(trainX, trainY, batch_size=batch_size, nb_epoch=nb_epoch,
          validation_data=(testX, testY), show_accuracy=True)
score, acc = model.evaluate(testX, testY, batch_size=batch_size,
                            show_accuracy=True)
print('Test score:', score)
print('Test accuracy:', acc)
