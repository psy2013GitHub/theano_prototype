#-*- encoding: utf8 -*-
__author__ = 'root'

import theano
import theano.tensor as T
from keras.layers.core import Layer, MaskedLayer
from keras import activations, initializations, constraints, regularizers

class Convolution(MaskedLayer):

    def __init__(self, filter_shape, border_mode = 'full',
                 init='glorot_uniform', inner_init='orthogonal',
                 activation='sigmoid', inner_activation='hard_sigmoid',
                 W_regularizer=None, activity_regularizer=None, W_constraint=None,
                 **kwargs):
        """
            输入，3D embedding 矩阵
            filter_shape: (n_filters, n_maps, filter_height, filter_width)
        """
        self.filter_shape = filter_shape
        self.n_filters, self.n_stacks, self.filter_word_dim_len, self.filter_embd_dim_len \
            = filter_shape
        self.n_maps = self.n_filters
        self.activation = activation
        self.border_mode = border_mode
        self.init = initializations.get(init)

        self.W_constraint = constraints.get(W_constraint)
        self.constraints = [self.W_constraint]

        self.W_regularizer = regularizers.get(W_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        super(Convolution, self).__init__(**kwargs)

    def build(self):
        self.W = self.init(self.filter_shape) # filter
        self.params = [self.W, ]

    def get_output(self, train=False):
        if len(self.input_shape)==3: # 从embedding输入，3D输入转为4D
            self.input = self.get_input(train=train)
            in_shape = self.input.shape
            X = T.reshape(self.input, [in_shape[0], 1, in_shape[1], in_shape[2]])
        else: # 从上个k-max输入
            X = self.get_input(train=train)
        conv_out = T.nnet.conv.conv2d(
                                X,
                                self.W,
                                border_mode = self.border_mode
        )
        return conv_out

    def output_conv_dim_shape(self):
        # 卷积层输出长宽
        seq_len = self.input_shape[-2]
        seq_width = self.input_shape[-1]
        if self.border_mode == 'full':
            output_word_dim_len = seq_len + self.filter_word_dim_len - 1
            output_embd_dim_len = seq_width + self.filter_embd_dim_len - 1
        elif self.border_mode == 'valid':
            output_word_dim_len = seq_len - self.filter_word_dim_len + 1
            output_embd_dim_len = seq_width - self.filter_embd_dim_len + 1
        else:
            raise ValueError('Unsupported conv mode: %s' % self.border_mode)
        return output_word_dim_len, output_embd_dim_len

    @property
    def output_shape(self):
        # 不管从哪里输入，长宽都是最后两位
        output_word_dim_len, output_embd_dim_len = self.output_conv_dim_shape()
        return self.input_shape[0], self.n_maps, output_word_dim_len, output_embd_dim_len

    def get_output_mask(self, train=False):
        mask, sents_len = self.get_input_mask(train=train)
        output_word_dim_len, output_embd_dim_len = self.output_conv_dim_shape()
        # 假设非mask部分数据在前
        def _step(k, max_seq_len, filter_word_dim_len):
            # 只保留最大长度句子长度序列
            return T.ones([max_seq_len,], dtype='int32') * T.lt(T.arange(max_seq_len), k + filter_word_dim_len) # 前面置为1，后面置为0
        mask, _ = theano.scan(_step,
            sequences=sents_len,
            non_sequences=[output_word_dim_len, self.filter_word_dim_len],
        )
        mask = T.reshape(mask, [mask.shape[0], 1, mask.shape[1], 1])
        return mask, sents_len

    def get_config(self):
        config = {
                    "name": self.__class__.__name__,
                    "output_shape": self.output_shape,
                    "init": self.init.__name__,
                 }
        base_config = super(Convolution, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
