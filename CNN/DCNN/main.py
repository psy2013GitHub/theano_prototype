#-*- encoding: utf8 -*-
__author__ = 'root'

import os
import sys
reload(sys)
sys.setdefaultencoding('utf8')

import numpy as np
import theano
from CNN.DCNN.nn_zoo import ExampleModel
from keras.utils.np_utils import to_categorical

#################################### data ####################################

# 加载数据
import CNN.DCNN.utils.stanfordSentimentTreebank as sst
sst_data_path = os.path.expanduser('~/NLPNN/CNN/DCNN/data/movie/')
skip_unknown_words = 0
shuffle_flag = 1
datatype = 5
if datatype == 5:
    # Fine-grained 5-class
    n_class = 5
elif datatype == 2:
    # Binary 2-class
    n_class = 2
else:
    raise ValueError('fuck')

# print "skip_unknown_words",skip_unknown_words
vocab, index2word, datasets, datasets_all_sentences, funcs = sst.load_stanfordSentimentTreebank_dataset(
    sst_data_path, normalize=True, skip_unknown_words=skip_unknown_words, datatype=datatype)
train_set, test_set, dev_set  = datasets
train_set_sentences, test_set_sentences, dev_set_sentences = datasets_all_sentences
get, sentence2ids, ids2sentence = funcs # 関数を読み込み
scores, sentences = zip(*train_set_sentences)
sentences = [[word for word in sentence.lower().split()] for sentence in sentences]

dev_unknown_count  = sum([unknown_word_count for score, (ids,unknown_word_count) in dev_set])
test_unknown_count = sum([unknown_word_count for score, (ids,unknown_word_count) in test_set])

# 让vocab下标从1开始，将0作为mask
for k, v in vocab.items():
    v.index += 1
index2word = [None,] + index2word

# 字典大小
vocab_size = len(vocab)

# 最大序列长度，同时让vocab下标从1开始
max_seq_len = 100
train_X = [[id+1 for id in ids] + [0,]*(max_seq_len-len(ids)) for score,(ids,unknown_word_count) in train_set]
train_Y = to_categorical([score for score,(ids,unknown_word_count) in train_set])

assert(sum([len(ids)!=max_seq_len for ids in train_X])==0)
assert(len(train_X)==len(train_Y))

train_X = np.array(train_X, dtype='int32')
train_Y = np.array(train_Y, dtype='int32')


test_X = [[id+1 for id in ids] + [0,]*(max_seq_len-len(ids)) for score,(ids,unknown_word_count) in test_set]
test_Y = to_categorical([score for score,(ids,unknown_word_count) in test_set])

assert(sum([len(ids)!=max_seq_len for ids in train_X])==0)
assert(len(test_X)==len(test_Y))

test_X = np.array(test_X, dtype='int32')
test_Y = np.array(test_Y, dtype='int32')


dev_X = [[id+1 for id in ids] + [0,]*(max_seq_len-len(ids)) for score,(ids,unknown_word_count) in dev_set]
dev_Y = to_categorical([score for score,(ids,unknown_word_count) in dev_set])

assert(sum([len(ids)!=max_seq_len for ids in train_X])==0)
assert(len(dev_X)==len(dev_Y))

dev_X = np.array(dev_X, dtype='int32')
dev_Y = np.array(dev_Y, dtype='int32')

#################################### model ####################################

print "train_size: %d, dev_size: %d, test_size: %d" %\
      (len(train_X), len(dev_X), len(test_X))
print "-"*30
print "vocab_size: %d, dev_unknown_words: %d, test_unknown_words: %d" %\
      (len(vocab), dev_unknown_count, test_unknown_count)



#theano.config.optimizer='None' # debug

# train model
embedding_size = 64
filter_shape_list = [
    (6, 1, 10, 1), # n_filters, n_stacks, width, height
    (12, 6, 7, 1), # stack_size 必须与上一个filter_num相同
#    (2, 2, 2, 1),
]
final_pool_size = 5

print len(train_X), len(train_Y)


clf = ExampleModel(
     max_seq_len,
     embedding_size,
     filter_shape_list,
     final_pool_size,
     vocab_size+1,
     n_class,
)
clf.fit(
    train_X, train_Y,
    batch_size=64,
    nb_epoch=100,
    verbose=1,
    callbacks=[],
    validation_split=0,
    validation_data=[dev_X, dev_Y],
    shuffle=True,
    show_accuracy=True,
    class_weight=None,
    sample_weight=None
)