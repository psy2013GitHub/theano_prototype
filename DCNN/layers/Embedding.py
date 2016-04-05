#-*- encoding: utf8 -*-
__author__ = 'momo'

import sys
reload(sys)
sys.setdefaultencoding('utf8')

'''
   重载keras embedding 使得能够通过mask传递句子长度
'''

import theano.tensor as T
from keras.layers.embeddings import Embedding as _Embedding

class Embedding(_Embedding):

    def __init__(self, *args, **kargs):
        super(Embedding, self).__init__(*args, **kargs)

    def get_output_mask(self, train=None):
        X = self.get_input(train)
        if not self.mask_zero:
            return None, None
        else:
            mask = T.ones_like(X) * (1 - T.eq(X, 0))
            sents_len = T.sum(T.neq(X, 0), axis=1) # 统计非0的数量即可
            return mask, sents_len