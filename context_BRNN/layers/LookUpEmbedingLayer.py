#-*- encoding: utf8 -*-
__author__ = 'root'

import theano
import theano.tensor as T
from keras.layers.core import Layer, MaskedLayer
from keras import initializations, constraints, regularizers
from utils.theano_util import make_shared


class LookUpEmbeddingLayer(MaskedLayer):

    input_ndim = 2

    def __init__(self, input_dim, EmbedMatrix, input_length=None,
                 init='uniform',
                 W_regularizer=None, activity_regularizer=None,
                 W_constraint=None,
                 mask_zero=False,
                 weights=None, **kwargs):
        '''

        :param input_dim: 字典大小
        :param input_length: max_seq_len最大序列长度
        :param EmbedMatrix: 已经计算好的 word2vec 矩阵，单词 * 嵌入
        '''
        self.input_dim = input_dim
        self.EmbedMatrix = theano.shared(value=EmbedMatrix)
        self.output_dim = EmbedMatrix.shape[1]
        self.init = initializations.get(init)
        self.input_length = input_length
        self.mask_zero = mask_zero

        self.W_constraint = constraints.get(W_constraint)
        self.constraints = [self.W_constraint]

        self.W_regularizer = regularizers.get(W_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.initial_weights = weights
        kwargs['input_shape'] = (self.input_dim,)
        super(LookUpEmbeddingLayer, self).__init__(**kwargs) # 在初始化里input_shape = [None, self.input_dim]

    def build(self):
        self.input = T.matrix('X', dtype='int32')
        self.regularizers = []
        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

    def get_output_mask(self, train=None):
        X = self.get_input(train)
        if not self.mask_zero:
            return None
        else:
            mask =  T.ones_like(X) * (1 - T.eq(X, 0)) # 创建二维mask矩阵, mask地方为0
            return mask

    @property
    def output_shape(self):
        return (self.input_shape[0], self.input_length, self.output_dim)

    def get_output(self, train=False):
        X = self.get_input(train)
        return self.EmbedMatrix[X] # 3 维张量，句子 * 词 * 词嵌入

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "input_dim": self.input_dim,
                  "output_dim": self.output_dim,
                  "init": self.init.__name__,
                  "input_length": self.input_length,
                  "mask_zero": self.mask_zero,
                  "activity_regularizer": self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  "W_regularizer": self.W_regularizer.get_config() if self.W_regularizer else None,
                  "W_constraint": self.W_constraint.get_config() if self.W_constraint else None}
        base_config = super(LookUpEmbeddingLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))