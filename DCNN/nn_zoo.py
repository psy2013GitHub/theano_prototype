#-*- encoding: utf8 -*-
from __future__ import division
import sys
reload(sys)
sys.setdefaultencoding('utf8')

import numpy as np
from keras.models import Sequential
from keras.regularizers import WeightRegularizer
from .layers.Embedding import Embedding
from keras.layers.core import Dense, Dropout
from .layers.Convolution import Convolution
from .layers.KMaxPooling import KMaxPooling
from .layers.Folding import Folding


def ExampleModel(max_seq_len, embedding_size, filter_shape_list, final_pool_size, dict_size, n_classes):
    model = Sequential()
    model.add(
        Embedding(
            dict_size, embedding_size,
            input_length=max_seq_len, mask_zero=True,
            W_regularizer=WeightRegularizer(l2=0.0001),
        )
    )
    model.add(
        Convolution(filter_shape_list[0], W_regularizer=WeightRegularizer(l2=0.00003))
    )
    model.add(Folding())
    model.add(KMaxPooling(final_pool_size, 2, 1, activation='tanh'))
    model.add(Convolution(filter_shape_list[1], W_regularizer=WeightRegularizer(l2=0.000003)))
    model.add(Folding())
    model.add(KMaxPooling(final_pool_size, 2, 2, activation='tanh', before_dense=True))
    model.add(Dropout(0.7))
    #model.add(Convolution(filter_shape_list[2]))
    #model.add(KMaxPooling(final_pool_size, 3, 3, before_dense=True)) # 最后一个k-max 记得设置before_dense
    model.add(Dense(n_classes, activation='softmax', W_regularizer=WeightRegularizer(l2=0.0001)))
    model.compile(
            loss='categorical_crossentropy',
            optimizer='adagrad',
            class_mode="categorical",
    )
    return model

