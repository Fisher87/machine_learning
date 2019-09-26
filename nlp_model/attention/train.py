# coding=utf8
import json
import time
import codecs
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from utils import *
from model import Model

train_file = './data/eng.train'
test_file  = './data/eng.testa'

left_train, right_train = parse(train_file)
left_test,  right_test  = parse(test_file)

label_x, id_x = np.unique(right_train, return_counts=True)

# gengrate train and test data;
ret = data_process(left_train, right_train, left_test, right_test)
word2idx, char2idx, tag2idx, train_X, train_Y, test_X, test_Y = ret
with codecs.open('./data/word2idx', 'w', 'utf8') as wf:
    wf.write(json.dumps(word2idx, ensure_ascii=False))
with codecs.open('./data/char2idx', 'w', 'utf8') as wf:
    wf.write(json.dumps(char2idx, ensure_ascii=False))
with codecs.open('./data/tag2idx', 'w', 'utf8') as wf:
    wf.write(json.dumps(tag2idx, ensure_ascii=False))

idx2word = {idx:word for word, idx in word2idx.items()}
idx2tag = {idx:tag for tag, idx in tag2idx.items()}

X_seq, Y_seq = to_train_seq(train_X, train_Y)
X_char_seq = generate_char_seq(X_seq, idx2word, char2idx)
X_seq_test, Y_seq_test = to_train_seq(test_X, test_Y)
X_char_seq_test = generate_char_seq(X_seq_test, idx2word, char2idx)

train_X, train_Y, train_char = X_seq, Y_seq, X_char_seq
test_X, test_Y, test_char = X_seq_test, Y_seq_test, X_char_seq_test

dim_word = 64
dim_char = 128
dropout = 0.8
learning_rate = 1e-3
hidden_size_char = 64
hidden_size_word = 64
num_layers = 2
batch_size = 32

model = Model(dim_word,dim_char,dropout,learning_rate,hidden_size_char,hidden_size_word,num_layers)

def train():
    with tf.Session() as sess:
        tf.global_variable_initializer()
        for e in range(3):
            lasttime = time.time()
            train_acc, train_loss, test_acc, test_loss = 0, 0, 0, 0
            #train data set
            pbar = tqdm(
                    range(0, len(train_X), batch_size), desc="train minibatch loop"
                    )
            for i in pbar:
                batch_x = train_X[i : min(i + batch_size, train_X.shape[0])]
                batch_char = train_char[i : min(i + batch_size, train_X.shape[0])]
                batch_y = train_Y[i : min(i + batchsize, train_X.shape[0])]
                acc, cost, _ = sess.run(
                        [model.accuracy, model.cost, model.optimizer],
                        feed_dict = {
                            model.word_ids: batch_x,
                            model.char_ids: batch_char,
                            model.labels  : batch_y
                            }
                        )
                assert not np.isnan(cost)
                train_loss += cost
                train_acc += acc
                pbar.set_postfix(cost=cost, accuracy=acc)

            #test data set
	    pbar = tqdm(
		    range(0, len(test_X), batch_size), desc = 'test minibatch loop'
		    )
	    for i in pbar:
		batch_x = test_X[i : min(i + batch_size, test_X.shape[0])]
		batch_char = test_char[i : min(i + batch_size, test_X.shape[0])]
		batch_y = test_Y[i : min(i + batch_size, test_X.shape[0])]
		acc, cost = sess.run(
		    [model.accuracy, model.cost],
		    feed_dict = {
			model.word_ids: batch_x,
			model.char_ids: batch_char,
			model.labels: batch_y
		    },
		)
		assert not np.isnan(cost)
		test_loss += cost
		test_acc += acc
		pbar.set_postfix(cost = cost, accuracy = acc)

            train_loss /= len(train_X)/batch_size
            train_acc  /= len(train_X)/batch_size
            test_loss  /= len(test_X)/batch_size
            test_acc   /= len(test_X)/batch_size
            print('time taken:', time.time() - lasttime)
            print(
                'epoch: %d, training loss: %f, training acc: %f, valid loss: %f, valid acc: %f\n'
                % (e, train_loss, train_acc, test_loss, test_acc)
            )

def test():
    pass

def eval():
    pass

def predict():
    pass
