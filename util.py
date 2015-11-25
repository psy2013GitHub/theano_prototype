#-*- encoding: utf8 -*-

import numpy as np
import theano
import theano.tensor as T

def make_shared(X, Y, borrow=True):
    shared_X = theano.shared(np.asarray(X, dtype=theano.config.floatX), borrow=True)
    shared_Y = theano.shared(np.asarray(Y, dtype=theano.config.floatX), borrow=True)
    return shared_X, T.cast(shared_Y, dtype='int32')
