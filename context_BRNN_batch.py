#-*- encoding: utf8 -*-

__author__ = ''

from collections import OrderedDict

import theano
import theano.tensor as T
from keras import activations, initializations, objectives
from keras.utils.theano_utils import shared_scalar, shared_zeros, alloc_zeros_matrix
from six.moves import range

class context_BRNN(object):

    def __init__(self, batch_size, nEmbed, nContext, nHidden, nClasses, learning_rate = 0.01, loss_func = 'binary_crossentropy',
                 init='glorot_uniform', inner_init='orthogonal', activation='sigmoid', weights=None,
                 truncate_gradient=-1, return_sequences=False, input_dim=None, input_length=None, **kwargs):
        self.batch_size = batch_size
        self.c = nContext 
        self.e = nEmbed
        self.H = nHidden 
        self.nClasses = nClasses
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.truncate_gradient = truncate_gradient
        self.activation = activations.get(activation)
        self.learning_rate=learning_rate
        self.loss_func = objectives.get(loss_func)
        self.initial_weights = weights

        self.build()

        X = T.dmatrix('X')
        Y = T.ivector('Y')
        nWords_lst = T.ivector('nWords_lst')

        def single_sent(X):
            single_sentX = T.concatenate([self.ew1, X, self.ewn], axis=0)

            def _forward_cl_step(ew_tm1, cl_tm1, Wl, Wsl):
               cl_t = self.activation(T.dot(cl_tm1, Wl) + T.dot(ew_tm1, Wsl))
               return T.cast(cl_t, theano.config.floatX)
            def _backward_cr_step(ew_tp1, cr_tp1, Wr, Wsr):
               cr_t = self.activation(T.dot(cr_tp1, Wr) + T.dot(ew_tp1, Wsr))
               return T.cast(cr_t, theano.config.floatX)
            def _cat_x_step(cl_tm1, ew_t, cr_tp1, W2, b2):
               x_t = T.concatenate([cl_tm1, ew_t, cr_tp1], axis=0)
               y2 = T.tanh(T.dot(x_t, W2) + b2)
               return T.cast(y2 , theano.config.floatX)
            
            CL, forward_updats = theano.scan(fn=_forward_cl_step,
                                  sequences=dict(input=single_sentX, taps=[-1]),
                                  outputs_info=dict(initial=self.cl1, taps=[-1]),
                                  non_sequences=[self.Wl, self.Wsl],
                                  strict=True,
            )
            CR, backward_updats = theano.scan(fn=_backward_cr_step,
                                  sequences=dict(input=single_sentX, taps=[-1]),
                                  outputs_info=dict(initial=self.crn, taps=[-1]),
                                  non_sequences=[self.Wr, self.Wsr],
                                  strict=True,
                                  go_backwards=True,
            ) 
            CR = CR[::-1] 
            Y2, x_updates = theano.scan(fn=_cat_x_step,
                                  sequences=[
                                     dict(input=CL, taps=[-1]),
                                     dict(input=single_sentX, taps=[0]),
                                     dict(input=CR, taps=[1]),
                                  ],
                                  non_sequences=[self.W2, self.b2],
                                  outputs_info=None,
                                  strict=True,
            )
            # max-pooling
            Y3 = T.max(Y2, axis=0) 
            # dense-activation
            Y4 = T.nnet.softmax(T.dot(Y3, self.W4) + self.b4)
      
            return Y4[:,0] 

        Y4_lst = []
        for i in range(self.batch_size):
            newX = X[nWords_lst[i]: nWords_lst[i+1], :]
            Y4_lst.append(single_sent(newX))
           
        predY = T.argmax(Y4_lst, axis=1)

        self.predict = theano.function(inputs=[X, nWords_lst], outputs=predY)

        cost = self.loss_func(Y, Y4_lst[:][0])
        self.cost = theano.function(inputs=[X, nWords_lst, Y], outputs=cost)

        gradients = T.grad(cost, self.params)
        train_updates = OrderedDict((p, p - learning_rate*g)
                                       for p, g in
                                       zip(self.params, gradients))
        self.train = theano.function(inputs=[X, nWords_lst, Y],
                                     outputs=cost,
                                     updates=train_updates,
        )

        self.error = theano.function(inputs=[X, nWords_lst, Y], outputs=(1 - T.eq(predY, Y)))


    def build(self):

        self.Wl  = self.init((self.c, self.c))
        self.Wr  = self.init((self.c, self.c))
        self.Wsl = self.init((self.e, self.c))
        self.Wsr = self.init((self.e, self.c))
        self.W2 = self.init((self.e + 2*self.c, self.H))
        self.b2 = shared_zeros((self.H),)
        self.cl1 = self.init([self.c])
        self.crn = self.init([self.c])
        self.ew1 = self.init([1, self.e])
        self.ewn = self.init([1, self.e])
        self.W4 = self.init((self.H, self.nClasses))
        self.b4 = self.init((self.nClasses,))

        self.params = [self.Wl, self.Wr, self.Wsl, self.Wsr, self.W2, self.b2, self.W4, self.b4, self.ew1, self.cl1, self.ewn, self.crn]

        if self.initial_weights is not None:
           self.set_weights(self.initial_weights)
           del self.initial_weights
