#-*- encoding: utf8 -*-
__author__ = 'root'

def load_data(txt_lable_dict, vocab_dict):
    X = []
    Y = []
    for txt, label in txt_lable_dict.iteritems():
        with open(txt, 'rb') as fid:
            for line in fid:
                tmp = line.strip().split()
                x = []
                if tmp:
                    for w in tmp:
                        idx = vocab_dict.get(w, None)
                        if idx:
                            x.append(idx)
                    if x:
                        X.append(x)
                        Y.append(label)
    assert len(X)==len(Y), 'len(X)!=len(Y)'
    return X, Y


def load_wordvector(txt):
    wordvectorMat = []
    vocabDict = dict()
    lineno = 0
    with open(txt, 'rb') as fid:
        for line in fid:
            if lineno == 0: # 跳过第一行
                lineno += 1
                continue
            tmp = line.strip().split()
            if tmp:
                word = tmp[0]
                vocabDict[word] = lineno # 从1开始计数
                if wordvectorMat is None or len(wordvectorMat) == 0:
                    wordvectorMat.append([0. for x in tmp[1:]]) # 一定要将第0行占住
                wordvectorMat.append([float(x) for x in tmp[1:]])
                lineno += 1
    return wordvectorMat, vocabDict



