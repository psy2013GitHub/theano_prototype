#-*- encoding: utf8 -*-
__author__ = ''

import theano
import theano.tensor as T
from keras.layers.core import Layer, MaskedLayer

class ElemMaxPool(MaskedLayer):
    '''
    按列取最大值
    :return:
    '''
    def __init__(self, output_dim, max_dim=1, **kwargs):
        self.output_dim = output_dim
        self.max_dim = max_dim
        super(ElemMaxPool,self).__init__(**kwargs)

    def get_output(self, train=False):
        X = self.get_input(train)
        # theano.printing.debugprint(X, print_type=True)
        return T.max(X, axis=self.max_dim, keepdims=False) # 默认降维

    @property
    def output_shape(self):
        return (self.input_shape[0], self.input_shape[2])

    def get_output_mask(self, train=False):
        return None# 设置get_output_mask为None便于连接其他非mask层

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "output_dim": self.output_dim,
                  }
        base_config = super(ElemMaxPool, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
