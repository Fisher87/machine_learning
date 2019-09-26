# coding=utf8
import numpy as np
import tensorflow as tf

def position_embedding(inputs, mask, embed_dim):
    """
    inputs: (B, L)
    mask:   (B, L)
    embed_dim: D
    return (B, L, D)
    """
    T = tf.shape(inputs)[1]   # L
    pos = tf.reshape(tf.range(0.0, tf.to_float(T), dtype=tf.float32), [-1, 1])  # (L, 1)
    i = np.arange(0, embed_dim, 2, np.float32)     # (D/2,)
    denom = np.reshape(np.power(10000.0, i/embed_dim), [1, -1])  # (1, D/2)
    enc = tf.expand_dims(tf.concat([tf.sin(pos/denom), tf.cos(pos/denom)], 1), 0) # (1, L, D)
    return tf.tile(enc, [tf.shape(inputs)[0], 1, 1]) * tf.expand_dims(tf.to_float(mask), -1)  # (B, L, D)

def multihead_attention(queries,
                        keys,
                        q_masks=None,
                        k_masks=None,
                        future_binding=None,
                        num_units=128,
                        num_heads=8
                        dropout_rate=0.1):
    """
    原始的transformer的项目代码(https://github.com/Kyubyong/transformer/blob/master/modules.py)
    中对于q_masks、k_mask是通过内部产生的(如下面mask函数定义),但对于这种生成方式是有问题的,因为
    在generate mask之前已经融合了pos encoding,所以tf.reduce_sum都是non-zero的.这也就导致通过使用
    tf.sign(tf.reduce_sum)来生成masks是有问题的(如#issue3[https://github.com/Kyubyong/transformer/issues/3])
    所描述的.因此正确做法应该是在add pos encoding之前就产生对应的masks, 再通过参数传入;

    process steps:
     1. generate Q, K, V;
     2. Q, K split to multi head groups;
     3. scaled dot product process to get score;
     4. normalized by d**0.5, d = num_units;
     5. softmax ; (attention map)
     6. product V;
     7. add;
    @param querys: [N, T_q, d_model]
           keys:   [N, T_k, d_model]
           q_mask: [N, T_q]
           k_mask: [N, T_k]
    """
    T_q = tf.shape(querys)[1]
    T_k = tf.shape(keys)[1]

    with tf.variable_scope("multihead_attention", reuse=tf.AUTO_REUSE):
        # generate Q, K, V  [linear projections]
        Q = tf.layers.dense(querys, num_units, use_bias=False)
        K = tf.layers.dense(keys, num_units, use_bias=False)
        V = tf.layers.dense(keys, num_units, use_bias=False)

        # split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)    # (N*h, T_q, d_model/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)    # (N*h, T_k, d_model/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)    # (N*h, T_k, d_model/h)

        # scaled_dot_product attention  ((Q*K^T)/d_model**0.5) * V
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))
        outputs /= tf.shape(K_)[-1] ** 0.5

        # key masking
        if not k_masks:
            # 有问题
            outputs = mask(ouputs, Q, K, type="key")
        else:
            outputs = new_mask(outputs, k_masks, type="key")

        # if future blinding masking
        if future_binding:
            outputs = mask(outputs, type="future")

        # get attention map
        outputs = tf.nn.softmax(outputs)
        attention = tf.transpose(outputs, [0, 2, 1])

        # query masking
        if not q_masks:
            # 有问题
            outputs = mask(outputs, Q, K, type="query")
        else:
            outputs = new_mask(outputs, q_masks, type="query")

        # dropout
        outputs = tf.layers.dropout(outputs, rate=dropout_rate)

        # weightd sum(context vectors)
        outputs = tf.matmul(ouputs, V)

    return outputs


def new_mask(inputs, num_heads, q_masks=None, k_masks=None, type=None):
    """
    inputs : [N, T_q, T_k]
    q_masks: [N, T_q]
    k_masks: [N, T_k]
    """
    padding_num = -2**32+1
    if type in ("k", "key", "keys") and k_masks:
        k_masks = tf.tile(k_masks, [num_heads, 1])
        k_masks = tf.tile(tf.expand_dims(k_masks, 1), [1, tf.shape(inputs)[1], 1])

        paddings = tf.ones_like(inputs) ** padding_num
        outputs = tf.where(tf.equal(k_masks, 0), paddings, inputs)
    elif type in ("q", "query", "queries") and q_masks:
        q_masks = tf.tile(q_masks, [num_heads, 1])
        q_masks = tf.tile(tf.expand_dims(q_masks, 1), [1, 1, tf.shape(inputs)[-1]])
        outputs*= q_masks
    elif type in ("f", "future", "right"):
        diag_vals = tf.ones_like(inputs[0, :, :]) # (T_q, T_k)
        tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense() #(T_q, T_k)
        masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(inputs)[0], 1, 1]) # (N, T_q, T_k)

        paddings = tf.ones_like(inputs) * padding_num
        outputs = tf.where(tf.equal(masks), paddings, inputs)
    else:
        print("check if you enterd type correctly!")

    return outputs

def mask(inputs, queries=None, keys=None, type=None):
    """
    inputs: (N, T_q, T_k)
    queries: (N, T_q, d)
    keys: (N, T_k, d)
    key/query mask: 对于key和query而言, 其输入有部分可能是padding的结果,
        对于这部分attention的结果应该进行mask;对于key mask通过给其一个很
        小的值(-2**32+1),使其在通过softmax进行计算attention map时score为0;
        而对于query而言无需经过softmax操作所以直接赋值为0即可;
    future mask: 在decode的过程中input只能看到之前的历史信息和encode信息，
        所以在decode self attention时对后面的信息进行mask.另外因为mask后
        要经过softmax进行处理,所以给其一个很小的值(-2**32+1);
    """
    padding_num = -2 ** 32 + 1
    if type in ("k", "key", "keys"):
        # generate mask
        masks = tf.sign(tf.reduce_sum(tf.abs(keys), axis=-1))   # (N, T_k)
        masks = tf.expand_dims(masks, 1)    # (N, 1, T_k)
        masks = tf.tile(masks, [1, tf.shape(querys)[1], 1])  # (N, T_q, T_k)

        # apply mask to inputs
        paddings = tf.ones_like(inputs) * padding_num
        outputs = tf.where(tf.equal(masks, 0), paddings, inputs)
    elif type in ("q", "query", "queries"):
        # generate mask
        masks = tf.sign(tf.reduce_sum(tf.abs(querys), axis=-1))  # (N, T_q)
        masks = tf.expand_dims(masks, -1)     # (N, T_q, 1)
        masks = tf.tile(masks, [1, 1, tf.shape(keys)[1]])   # (N, T_q, T_k)

        outpus = inputs * masks
    elif type in ("f", "future", "right"):
        diag_vals = tf.ones_like(inputs[0, :, :]) # (T_q, T_k)
        tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense() #(T_q, T_k)
        masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(inputs)[0], 1, 1]) # (N, T_q, T_k)

        paddings = tf.ones_like(inputs) * padding_num
        outputs = tf.where(tf.equal(masks), paddings, inputs)
    else:
        print("check if you enterd type correctly!")

    return outputs

def pointwise_feedforward(inputs, hidden_units, activation=None):
    outputs = tf.layers.dense(inputs, 4*hidden_units, activation=activation)
    outputs = tf.layers.dense(outputs, hidden_units, activation=None)

    # Residual connection , 残差
    outputs += inputs
    outputs = layer_norm(outputs)
    return outputs

def layer_norm(inputs, epsilon=1e-8):
    #TODO
    mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
    normalized = (inputs - mean) / (tf.sqrt(variance + epsilon))

    params_shape = inputs.get_shape()[-1:]
    gamma = tf.get_variable('gamma', params_shape, tf.float32, tf.ones_initializer())
    beta = tf.get_variable('beta', params_shape, tf.float32, tf.zeros_initializer())

    outputs = gamma * normalized + beta
    return outputs

