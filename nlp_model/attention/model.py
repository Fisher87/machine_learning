# coding=utf8

import numpy as np
import tensorflow as tf

from modules import position_embedding
from modules import multihead_attention
from modules import pointwise_feedforward

"""
参数维度说明:
    dim_word : W
    dim_char : C
    hidden_size_word: w_
    hidden_size_char: c_
    batchsize: B
    SeqLength: L
    wordLength: l
    word_dict_length : V
    char_dict_length : S

"""

class Model(object):
    def __init__(self, dim_word,
                       dim_char,
                       dropout,
                       learning_rate,
                       hidden_size_char,
                       hidden_size_word,
                       num_blocks=2,
                       num_heads=8,
                       min_freq=50,
                       config=None):
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None], name="word_input")         # (B, L)
        self.char_ids = tf.placeholder(tf.int32, shape=[None, None, None], name="char_input")   # (B, L, l)
        self.labels = tf.placeholder(tf.int32, shape=[None], name="labels")
        self.maxlen = tf.shape(self.word_ids)[1]    # L
        batchsize   = tf.shape(self.word_ids)[0]    # B
        self.lengths = tf.count_nonzero(self.word_ids, 1)
        self.config = config

        self.word_embedding = tf.Variable(
                            tf.truncated_normal([len(word2idx), dim_word], stddev=1.0/np.sqrt(dim_word)))   # (V, W)
        self.char_embedding = tf.Variable(
                            tf.truncated_normal([len(char2idx), dim_char], stddev=1.0/np.sqrt(dim_char)))   # (S, C)

        word_embedded = tf.nn.embedding_lookup(self.word_embedding, self.word_ids, name="word_embedding")   # (B, L, W)
        char_embedded = tf.nn.embedding_lookup(self.char_embedding, self.char_ids, name="char_embedding")   # (B, L, l, C)
        s = tf.shape(char_embedded)    # [B, L, l, C]
        char_embedded = tf.reshape(char_embedded, shape=[s[0]*s[1], s[-2], dim_char])   # (B*L, l, C)
        reshape_char = tf.reshape(self.char_ids, shape=[s[0]*s[1], s[-2]])              # (B*L, l)
        char_masked = tf.sign(reshape_char)    # (B*L, l)
        #add pos embedding
        char_embedded += position_embedding(reshape_char, char_mask, dim_char)   #(B*L, l, C)

        #add attention
        for i in range(num_blocks):
            with tf.variable_scope("char_%d" %i, reuse=tf.AUTO_REUSE):
                char_embedded = multihead_attention(queries=char_embedded,
                                                    keys  =char_embedded,
                                                    q_masks=char_masked,
                                                    k_masks=char_masked,
                                                    future_binding=False,
                                                    num_units = dim_char,
                                                    num_heads = num_heads
                                                    )
            with tf.variable_scope("char_feedforward_%d" %i, reuse=tf.AUTO_REUSE):
                char_embedded = pointwise_feedforward(char_embedded, dim_char, activation=tf.nn.relu)

        # char_embedded[:, -1]    (B*L, C)
        # output (B, L, 2*c_)
        output = tf.reshape(
                char_embedded[:, -1], shape=[s[0], s[1], 2*hidden_size_char]
                )

        word_embedded = tf.concat([word_embedded, output], axis=-1)    # (B, L, W+2*c_)
        word_embedded = tf.layers.dense(word_embedded, dim_char)       # (B, L, C)
        de_masks = tf.sign(self.word_ids)
        word_embedded += position_embedding(self.word_ids, word_masks, dim_char)

        for i in range(num_blocks):
            with tf.variable_scope("word_char_%d" %i, reuse=tf.AUTO_REUSE):
                decoder_embedded = multihead_attention(queries=word_embedded,
                                                       keys =word_embedded,
                                                       q_masks=de_masks,
                                                       k_masks=de_masks,
                                                       future_binding=True,
                                                       num_units= dim_char,
                                                       num_heads= num_heads)
            with tf.variable_scope("word_char_attention_%d" %i, reuse=tf.AUTO_REUSE):
                decoder_embedded = multihead_attention(queries=decoder_embedded,
                                                       keys = output,
                                                       q_masks = de_masks,
                                                       k_masks = de_masks,
                                                       future_binding=False,
                                                       num_units= dim_char,
                                                       num_heads= num_heads)
            with tf.variable_scope("word_feedforward_%d" %i, reuse=tf.AUTO_REUSE):
                decoder_embedded = pointwise_feedforward(decoder_embedded, dim_char, activation=tf.nn.relu)

        logits = tf.layers.dense(decoder_embedded, idx2tag_length)
        y_t = self.labels
        log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(logits, y_t, self.lengths)

        self.cost = tf.reduce_mean(-log_likelihood)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)
        mask = tf.sequence_mask(self.lengths, maxlen=self.maxlen)
        self.tags_seq, tags_score = tf.contrib.crf.crf_decode(logits, transition_params, self.lengths)
        self.tags_seq = tf.identity(self.tags_seq, name="logits")

        y_t = tf.cast(y_t, tf.int32)
        self.prediction = tf.boolean_mask(self.tags_seq, mask)
        mask_label = tf.boolean_mask(y_t, mask)

        correct_pred = tf.equal(self.prediction, mask_label)
        correct_index= tf.cast(correct_pred, tf.float32)
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))





