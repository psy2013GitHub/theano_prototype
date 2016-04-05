#-*- encoding: utf8 -*-
from __future__ import division

import numpy as np
import theano
import theano.tensor as T
from keras.layers.core import Layer, MaskedLayer
from keras import activations, initializations

class KMaxPooling(MaskedLayer):
    input_ndim = 4 # 输入必须为4D: nSamples * nMaps * Height * Width
    def __init__(self, final_pool_size, total_conv_levels, curr_conv_level,
                 before_dense=False, activation='sigmoid', init='glorot_uniform'):
        '''
        :param pooling_size_lst: 每个句子对应的pooling_size列表
        :param before_dense: 是不是在full connected layer 前
        :return:
        '''
        self.before_dense = before_dense
        self.activation = activations.get(activation) # 文中non-linear function g
        self.init = initializations.get(init)
        self.final_pool_size, self.total_conv_levels, self.curr_conv_level = \
                final_pool_size, total_conv_levels, curr_conv_level
        super(KMaxPooling, self).__init__()

    def dynamic_pool_size(self, final_pool_size, total_conv_levels, curr_conv_level, sents_len_tensor):
        tmp = (total_conv_levels - curr_conv_level) / total_conv_levels
        return T.maximum(
                final_pool_size,
                T.ceil(tmp * sents_len_tensor)
        ).astype('int32') # 整数

    def get_output_mask(self, train=False):
        if self.before_dense: # 全连接层 前直接None
            return None
        # mask 与 前一层 mask 无关
        _, sents_len = self.get_input_mask(train=train)
        def _step(k, max_seq_len):
            # 只保留最大长度句子长度序列
            return T.ones([max_seq_len,], dtype='int32') * T.lt(T.arange(max_seq_len), k)
        max_seq_len = T.max(sents_len)
        mask, _ = theano.scan(_step,
            sequences=sents_len,
            non_sequences=max_seq_len,
            #outputs_info=None,
        )
        return mask, sents_len

    def build(self):
        n_samples, n_maps, height, width = self.input_shape
        self.b = self.init([n_maps,])
        self.params = [self.b,]

    def get_output(self, train=False):
        X = self.get_input(train)
        mask, sents_len = self.get_input_mask(train)
        k_list = self.dynamic_pool_size(
            self.final_pool_size, self.total_conv_levels, self.curr_conv_level, sents_len
        )
        max_seq_len = T.max(k_list)

        if mask is None:
            masked_data = X
        else:
            # mask = mask.dimshuffle(0, 1, 'x', 'x')
            # 将mask对应数变为-inf
            masked_data = T.switch(T.eq(mask, 0), -np.inf, X)

        # 计算每个句子对应得k

        # 对每句k_max
        def _step(x, k, max_seq_len):
            tmp = x[
                T.arange(x.shape[0])[:, np.newaxis, np.newaxis],
                T.sort(T.argsort(x, axis=1)[:, -k:, :], axis=1),
                T.arange(x.shape[2])[np.newaxis, np.newaxis,:],
            ]
            return T.concatenate([tmp, T.zeros([x.shape[0], max_seq_len-k, x.shape[2]])], axis=1)
        pooled_data, _ = theano.scan(_step,
            sequences=[masked_data, k_list],
            non_sequences=max_seq_len,
            outputs_info=None,
        )

        # +b & non-linear function g
        Y = self.activation(
            pooled_data + self.b.dimshuffle('x', 0, 'x', 'x')
        )

        # 在full connected layer 前就flatten Y
        if self.before_dense:
            Y = Y.flatten(2)
        return Y

    def fix_k_max(self, k, masked_data):
        # @ref: https://github.com/fchollet/keras/issues/373
        result = masked_data[
            T.arange(masked_data.shape[0]).dimshuffle(0, "x", "x"),
            T.sort(T.argsort(masked_data, axis=1)[:, -k:, :], axis=1),
            T.arange(masked_data.shape[2]).dimshuffle("x", "x", 0)
        ]
        return result

    @property
    def output_shape(self):
        mask, sents_len = self.get_input_mask()
        k_list = self.dynamic_pool_size(
            self.final_pool_size, self.total_conv_levels, self.curr_conv_level, sents_len
        )
        max_seq_len = T.max(k_list)

        if not self.before_dense:
            return self.input_shape[:2] + (max_seq_len, self.input_shape[3])
        else:
            # fixme 因为最后Dense层必须接受非tensor shape, 同时 最后一层理论上必然是final_pool_size，但是此处最好加上assert
            max_seq_len = self.final_pool_size
            return self.input_shape[0], self.input_shape[1] * max_seq_len * self.input_shape[3]

    def get_config(self):
        return {"name" : self.__class__.__name__, "pooling_size" : self.pooling_size}


def unittest(x, mask, k_list):
    X = T.tensor3('X')
    Mask = T.tensor3('Mask')
    inst = KMaxPooling(k_list, 1, 2) # fixme 修改新滴unittest函数
    inst.input = X
    # 打补丁
    def get_input_mask(train=False):
        return Mask
    inst.get_input_mask = get_input_mask
    Y = inst.get_output()
    f = theano.function(inputs=[X, Mask], outputs=Y)
    return f(x, mask)

