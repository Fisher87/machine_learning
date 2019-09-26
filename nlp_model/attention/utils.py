# coding=utf8

import numpy as np

word2idx = {'PAD': 0,'NUM':1,'UNK':2}
tag2idx = {'PAD': 0}
char2idx = {'PAD': 0}
word_idx = 3
seq_len = 50

def parse(file):
    with open(file) as rf:
        texts = rf.read().split('\n')
    left, right = [], []
    for text in texts:
        if not text:continue
        if "-DOCSTART-" in text or not len(text) :
            continue
        splitted = text.split()
        left.append(splitted[0])
        right.append(splitted[1])
    return left, right

# generate data func;
def data_process(train_texts, train_labels, test_texts, test_labels):
    train_X, train_Y, test_X, test_Y = [], [], [], []
    for no, text in enumerate(train_texts):
        text = text.lower()
        tag = train_labels[no]
        for c in text:
            if c not in char2idx:
                char2idx[c] = len(char2idx)+1
        if tag not in tag2idx:
            tag2idx[tag] = len(tag2idx)+1
        train_Y.append(tag2idx[tag])
        if text not in word2idx:
            word2idx[text] = len(word2idx)+1
        train_X.append(word2idx[text])

    for no, text in enumerate(test_texts):
        text = text.lower()
        tag = test_labels[no]
        for c in text:
            if c not in char2idx:
                char2idx[c] = len(char2idx)+1
        if tag not in tag2idx:
            tag2idx[tag] = len(tag2idx)+1
        test_Y.append(tag2idx[tag])
        if text not in word2idx:
            word2idx[text] = len(word2idx)+1
        test_X.append(word2idx[text])
    return (word2idx, char2idx, tag2idx, train_X, train_Y, test_X, test_Y)

def iter_seq(x):
    k = np.array([x[i: i+seq_len] for i in range(0, len(x)-seq_len, 1)])
    return k

def to_train_seq(*args):
    return [iter_seq(x) for x in args]

def generate_char_seq(batch, idx2word, char2idx):
    x = [[len(idx2word[i]) for i in k] for k in batch]
    maxlen = max([j for i in x for j in i])
    temp = np.zeros((batch.shape[0],batch.shape[1],maxlen),dtype=np.int32)
    for i in range(batch.shape[0]):
        for k in range(batch.shape[1]):
            for no, c in enumerate(idx2word[batch[i,k]]):
                temp[i,k,-1-no] = char2idx[c]
    return temp
