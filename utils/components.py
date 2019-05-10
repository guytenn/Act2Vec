import sys
sys.path.insert(0, '.')

import tensorflow as tf
from tensorflow.contrib import rnn


class batch_norm(object):
    """
    Class for creating a batch normalization layer
    """
    def __init__(self, epsilon=1e-5, momentum=0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x,
                                            decay=self.momentum,
                                            updates_collections=None,
                                            epsilon=self.epsilon,
                                            scale=True,
                                            is_training=train,
                                            scope=self.name)



def convReluTopKConcat(input, numOfFilters, kernelWidthVec, TopK, axisToConcat, name):
    '''
    Conv layer followed by Relu, followed by topK pooling and concatenation
    '''

    x = None

    kernelHeight = input.get_shape()[1]
    for i in range(len(kernelWidthVec)):
        xtmp = conv2d(input, numOfFilters, kernelHeight, kernelWidthVec[i],
                           name=name + str(kernelWidthVec[i]))
        batchNorm = batch_norm(name=name + 'batchNorm' + str(kernelWidthVec[i]))
        xtmp = batchNorm(xtmp)
        xtmp = tf.nn.relu(xtmp)
        xtmp = dynamicTopK(input=xtmp, axis=2, TopK=TopK)
        xtmp = tf.transpose(xtmp, [0, 3, 2, 1])
        if x is not None:
            x = tf.concat([x, xtmp], axis=axisToConcat)
        else:
            x = xtmp
    x = tf.squeeze(x)
    return x

def dynamicTopK(input, axis, TopK):
    """
    max pooling layer with option to use top K values instead of regular max pool
    """
    x = input
    if TopK == 1:
        inds = [0, axis, 2, 3]
        inds[axis] = 1
        x = tf.transpose(x, inds)
        x = tf.nn.max_pool(x, ksize=[1, x.get_shape()[1], 1, 1], strides=[1, 1, 1, 1], padding="VALID")
        x = tf.transpose(x, inds)
    else:
        inds = [0, 1, 2, axis]
        inds[axis] = 3
        x = tf.transpose(x, inds)
        x, _ = tf.nn.top_k(x, TopK)
        x = tf.transpose(x, inds)
    return x



def conv2d(input, outputDim, kernelHeight, kernelWidth,
           strides=[1, 1], stddev=0.02, padding="VALID", name="conv2d"):
    """
    2d convolution layer
    """
    with tf.variable_scope(name):
        W = tf.get_variable('W', [kernelHeight, kernelWidth, input.get_shape()[-1], outputDim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        b = tf.get_variable('b', [outputDim], initializer=tf.constant_initializer(0.0))

        conv = tf.nn.conv2d(input, W, strides=[1, strides[0], strides[1], 1], padding=padding)
        # conv = tf.reshape(tf.nn.bias_add(conv, b), conv.get_shape())

        return conv + b


def linear(input, outputDim, withBias=True, stddev=0.02, name="linear"):
    """
    Fully-connected layer
    """
    x = tf.contrib.layers.flatten(input)
    with tf.variable_scope(name):
        W = tf.get_variable("W", [x.get_shape()[1], outputDim], tf.float32,
                            tf.random_normal_initializer(stddev=stddev))
        if withBias:
            b = tf.get_variable("bias", [outputDim], initializer=tf.constant_initializer(0))
            res = tf.matmul(x, W) + b
        else:
            res = tf.matmul(x, W)
        return res


def multiLinear(input, dims, sttdev, names, activation=tf.nn.tanh):
    """
    MLP with multiple linear layers (default activation is tanh)
    """
    x = input
    for i in xrange(len(dims)):
        x = linear(input=x,
                   outputDim=dims[i],
                   stddev=sttdev[i],
                   name=names[i])
        x = activation(x)
    return x


def LSTM(input,
         outputDim,
         BiDirectional=False,
         onlyLastOutput=True,
         name='LSTM',
         seqLength=None,
         returnStates=False,
         activation=tf.tanh,
         reuse=True):
    """
   Bi-directional long short term memory layer
   Input must be of shape [batch_size, max_time, ...]
   Output is of shape [batch_size, max_time, cell.output_size]
   """
    with tf.variable_scope(name):
        # Define lstm cells with tensorflow
        # Forward direction cell
        x = input

        lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(outputDim, forget_bias=1.0, activation=activation, reuse=reuse)
        if BiDirectional:
            # Backward direction cell
            lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(outputDim, forget_bias=1.0, activation=activation, reuse=reuse)

            outputs, states = \
                tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell,
                                                cell_bw=lstm_bw_cell,
                                                inputs=x,
                                                sequence_length=seqLength,
                                                dtype="float")
            outputs = tf.concat([outputs[0], outputs[1]], axis=2)
            states = tf.concat([states[0][0], states[1][0]], axis=1)
        else:
            outputs, states = \
                tf.nn.dynamic_rnn(cell=lstm_fw_cell,
                                  inputs=x,
                                  sequence_length=seqLength,
                                  dtype="float")
            states = states[0]
        if onlyLastOutput:
            if seqLength is None:
                outputs = outputs[:, -1, :]# tf.squeeze(outputs[:, -1, :])
            else:
                batch_range = tf.range(tf.shape(outputs)[0])
                indices = tf.stack([batch_range, seqLength - 1], axis=1)
                outputs = tf.gather_nd(outputs, indices)
                # outputs = tf.squeeze(outputs[:, seqLength, :])

        if returnStates:
            return outputs, states
        else:
            return outputs


def factorizationMachine(input, k=8, stddev=0.02, name="FM"):
    """
    Implementation of a factorization machine
    """
    def ord2(slice, V):
        slice = tf.expand_dims(slice, axis=1)
        order2 = tf.matmul(tf.matmul(slice, V, transpose_a=True), slice)
        return tf.squeeze(order2*tf.ones_like(slice))

    x = tf.contrib.layers.flatten(input)
    with tf.variable_scope(name):
        W = tf.get_variable("W", [x.get_shape()[1], 1], tf.float32,
                            tf.random_normal_initializer(stddev=stddev))
        order1 = tf.matmul(x, W)

        V1 = tf.get_variable("V1", [x.get_shape()[1], k], tf.float32,
                             tf.random_normal_initializer(stddev=stddev))
        V2 = tf.get_variable("V2", [x.get_shape()[1], k], tf.float32,
                             tf.random_normal_initializer(stddev=stddev))
        V = tf.matmul(V1, V2, transpose_b=True)
        V = tf.matrix_band_part(V, 1, 0) # upper triangular part (without diagonal)

        # batchSize = x.get_shape()[0]
        # slice = tf.expand_dims(x[0], axis=1)
        # order2 = tf.matmul(tf.matmul(slice, V, transpose_a=True), slice)
        order2 = tf.scan(lambda a, slice: ord2(slice, V), x)[:, 0]
        order2 = tf.expand_dims(order2, axis=1)
        # for i in xrange(1, batchSize):
        #     slice = tf.expand_dims(x[i], axis=1)
        #     order2 = tf.concat([order2, tf.matmul(tf.matmul(slice, V, transpose_a=True), slice)], axis=0)

        b = tf.get_variable("bias", [1], initializer=tf.constant_initializer(0))

        return order1 + order2 + b




def Global_Attention(inputs, attention_size, time_major=False, return_alphas=False):
    """
    Attention mechanism layer which reduces RNN/Bi-RNN outputs with Attention vector.
    The idea was proposed in the article by Z. Yang et al., "Hierarchical Attention Networks
     for Document Classification", 2016: http://www.aclweb.org/anthology/N16-1174.
    Args:
        inputs: The Attention inputs.
            Matches outputs of RNN/Bi-RNN layer (not final state):
                In case of RNN, this must be RNN outputs `Tensor`:
                    If time_major == False (default), this must be a tensor of shape:
                        `[batch_size, max_time, cell.output_size]`.
                    If time_major == True, this must be a tensor of shape:
                        `[max_time, batch_size, cell.output_size]`.
                In case of Bidirectional RNN, this must be a tuple (outputs_fw, outputs_bw) containing the forward and
                the backward RNN outputs `Tensor`.
                    If time_major == False (default),
                        outputs_fw is a `Tensor` shaped:
                        `[batch_size, max_time, cell_fw.output_size]`
                        and outputs_bw is a `Tensor` shaped:
                        `[batch_size, max_time, cell_bw.output_size]`.
                    If time_major == True,
                        outputs_fw is a `Tensor` shaped:
                        `[max_time, batch_size, cell_fw.output_size]`
                        and outputs_bw is a `Tensor` shaped:
                        `[max_time, batch_size, cell_bw.output_size]`.
        attention_size: Linear size of the Attention weights.
        time_major: The shape format of the `inputs` Tensors.
            If true, these `Tensors` must be shaped `[max_time, batch_size, depth]`.
            If false, these `Tensors` must be shaped `[batch_size, max_time, depth]`.
            Using `time_major = True` is a bit more efficient because it avoids
            transposes at the beginning and end of the RNN calculation.  However,
            most TensorFlow data is batch-major, so by default this function
            accepts input and emits output in batch-major form.
        return_alphas: Whether to return attention coefficients variable along with layer's output.
            Used for visualization purpose.
    Returns:
        The Attention output `Tensor`.
        In case of RNN, this will be a `Tensor` shaped:
            `[batch_size, cell.output_size]`.
        In case of Bidirectional RNN, this will be a `Tensor` shaped:
            `[batch_size, cell_fw.output_size + cell_bw.output_size]`.
    """

    if isinstance(inputs, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        inputs = tf.concat(inputs, 2)

    if time_major:
        # (T,B,D) => (B,T,D)
        inputs = tf.array_ops.transpose(inputs, [1, 0, 2])

    inputs_shape = inputs.shape
    sequence_length = inputs_shape[1].value  # the length of sequences processed in the antecedent RNN layer
    hidden_size = inputs_shape[2].value  # hidden size of the RNN layer

    # Attention mechanism
    W_omega = tf.get_variable('W_omega',
                              [hidden_size, attention_size],
                              tf.float32,
                              initializer=tf.truncated_normal_initializer(stddev=0.1))
    b_omega = tf.get_variable('b_omega',
                              [attention_size],
                              tf.float32,
                              initializer=tf.truncated_normal_initializer(stddev=0.1))
    u_omega = tf.get_variable('u_omega',
                              [attention_size],
                              tf.float32,
                              initializer=tf.truncated_normal_initializer(stddev=0.1))

    v = tf.tanh(tf.matmul(tf.reshape(inputs, [-1, hidden_size]), W_omega) + tf.reshape(b_omega, [1, -1]))
    vu = tf.matmul(v, tf.reshape(u_omega, [-1, 1]))
    exps = tf.reshape(tf.exp(vu), [-1, sequence_length])
    alphas = exps / tf.reshape(tf.reduce_sum(exps, 1), [-1, 1])

    # Input is reduced with attention vector
    output = tf.reduce_sum(inputs * tf.reshape(alphas, [-1, sequence_length, 1]), 1)

    if not return_alphas:
        return output
    else:
        return output, alphas



def intraLevelAttention(inputs, intraWordEmbeddingDim, name='intra'):

    with tf.variable_scope(name):
        inputs_shape = inputs.shape
        batchSize = inputs_shape[0].value
        nWords = inputs_shape[1].value
        words = tf.split(inputs, nWords, 1)

        F = [tf.tanh(linear(w, intraWordEmbeddingDim, stddev=0.02, name="intraLinear_"+str(i))) for i, w in enumerate(words)]
        newWords = [None for _ in xrange(nWords)]
        for i, w in enumerate(words):
            fij = [tf.expand_dims(tf.reduce_sum(tf.multiply(F[i], F[j]), axis=1), 1) for j in xrange(nWords)]
            # distBias = [abs(i-j) for j in xrange(nWords)]
            distBias = [tf.get_variable("distBias"+str(i)+str(j),
                                        [batchSize, 1],
                                        tf.float32,
                                        tf.constant_initializer(0))
                        for j in xrange(nWords)]
            logits = [tf.tanh(fij[j] + distBias[j]) for j in xrange(nWords)]
            att = [tf.exp(logits[j]) for j in xrange(nWords)]
            attSum = tf.expand_dims(tf.reduce_sum(tf.concat(att, 1), axis=1), 1)
            attNormalized = [val / attSum for val in att]
            attendedWords = [tf.expand_dims(attNormalized[0], 2)[j] * words[j] for j in xrange(nWords)]
            newWords[i] = reduce((lambda x, y: x+y), attendedWords)
        output = tf.concat(newWords, axis=1)
    return output, attNormalized


def ARSG(inputs, seqLen=None, attention_size=None, reuse=True, name='ARSG'):
    '''
        Attention-based recurrent sequence generator
        https://arxiv.org/pdf/1506.07503.pdf
        https://arxiv.org/pdf/1508.04395.pdf
    '''
    with tf.variable_scope(name):
        att = LSTM(input=inputs,
                   outputDim=attention_size,
                   BiDirectional=False,
                   onlyLastOutput=False,
                   name='ARSG_LSTM',
                   seqLength=seqLen,
                   returnStates=False,
                   activation=tf.tanh,
                   reuse=reuse)
        W = tf.get_variable("W", [attention_size, inputs.shape[1].value], tf.float32,
                            tf.random_normal_initializer(stddev=0.02))
        attList = tf.unstack(att)
        for i in xrange(len(attList)):
            attList[i] = tf.exp(tf.matmul(attList[i], W))
            attList[i] / tf.reduce_sum(attList[i], axis=1)
        attNormalized = tf.concat([tf.expand_dims(a, axis=0) for a in attList], axis=0)
        outputs = tf.matmul(attNormalized, inputs)
    return outputs, attNormalized



def reshapeFoldTensor(tensor, endAxis):
    tensorShape = tensor.get_shape().as_list()
    while len(tensorShape) > endAxis + 1:
        newShape = tensorShape[0:-1]
        newShape[-1] *= tensorShape[-1]
        tensor = tf.reshape(tensor, newShape)
        tensorShape = tensor.get_shape().as_list()
    return tensor




def projectEmbeddings(embeddings, projectionDim, reuse=False):
    """
    Project word embeddings into another dimensionality

    :param embeddings: embedded sentence, shape (batch, time_steps,
        embedding_size)
    :param reuse: reuse weights in internal layers
    :return: projected embeddings with shape (batch, time_steps, num_units)
    """
    timeSteps = tf.shape(embeddings)[1]
    embeddingDim = embeddings.shape[2].value
    embeddings2d = tf.reshape(embeddings, [-1, embeddingDim])

    with tf.variable_scope('projection', reuse=reuse):
        initializer = tf.random_normal_initializer(0.0, 0.1)
        weights = tf.get_variable('weights',
                                  [embeddingDim, projectionDim],
                                  initializer=initializer)

        projected = tf.matmul(embeddings2d, weights)

    projected3d = tf.reshape(projected, tf.stack([-1, timeSteps, projectionDim]))
    return projected3d

def reluLayer(inputs, weights, bias, dropout):
    """
    Apply dropout to the inputs, followed by the weights and bias,
    and finally the relu activation

    :param inputs: 2d tensor
    :param weights: 2d tensor
    :param bias: 1d tensor
    :return: 2d tensor
    """
    inputs = tf.cast(inputs, "float")
    after_dropout = tf.nn.dropout(inputs, dropout)
    raw_values = tf.nn.xw_plus_b(after_dropout, weights, bias)
    return tf.nn.relu(raw_values)

def feedForwardWithRelus(inputs, num_input_units, scope, num_units,
                         reuse_weights=False, initializer=None, dropout=0.8):
    """
    Apply two feed forward layers with self.num_units on the inputs.
    :param inputs: tensor in shape (batch, time_steps, num_input_units)
        or (batch, num_units)
    :param num_input_units: a python int
    :param reuse_weights: reuse the weights inside the same tensorflow
        variable scope
    :param initializer: tensorflow initializer; by default a normal
        distribution
    :param num_units: list of length 2 containing the number of units to be
        used in each layer
    :return: a tensor with shape (batch, time_steps, num_units)
    """
    rank = len(inputs.get_shape())
    if rank == 3:
        time_steps = tf.shape(inputs)[1]

        # combine batch and time steps in the first dimension
        inputs2d = tf.reshape(inputs, tf.stack([-1, num_input_units]))
    else:
        inputs2d = inputs

    initializer = initializer or tf.random_normal_initializer(0.0, 0.1)

    scope = scope or 'feedforward'
    with tf.variable_scope(scope, reuse=reuse_weights):
        with tf.variable_scope('layer1'):
            shape = [num_input_units, num_units]
            weights1 = tf.get_variable('weights', shape,
                                       initializer=initializer)
            zero_init = tf.zeros_initializer()
            bias1 = tf.get_variable('bias', shape=num_units,
                                    dtype=tf.float32,
                                    initializer=zero_init)

        with tf.variable_scope('layer2'):
            shape = [num_units, num_units]
            weights2 = tf.get_variable('weights', shape,
                                       initializer=initializer)
            zero_init = tf.zeros_initializer()
            bias2 = tf.get_variable('bias', shape=num_units,
                                    dtype=tf.float32,
                                    initializer=zero_init)

        # relus are (time_steps * batch, num_units)
        relus1 = reluLayer(inputs2d, weights1, bias1, dropout)
        relus2 = reluLayer(relus1, weights2, bias2, dropout)

    if rank == 3:
        output_shape = tf.stack([-1, time_steps, num_units])
        relus2 = tf.reshape(relus2, output_shape)

    return relus2

def get_distance_biases(time_steps, distance_biases, reuse_weights=False):
    """
    Return a 2-d tensor with the values of the distance biases to be applied
    on the intra-attention matrix of size sentence_size

    :param time_steps: tensor scalar
    :return: 2-d tensor (time_steps, time_steps)
    """
    with tf.variable_scope('distance-bias', reuse=reuse_weights):
        # this is d_{i-j}
        distance_bias = tf.get_variable('dist_bias', [distance_biases],
                                        initializer=tf.zeros_initializer())

        # messy tensor manipulation for indexing the biases
        r = tf.range(0, time_steps)
        r_matrix = tf.tile(tf.reshape(r, [1, -1]),
                           tf.stack([time_steps, 1]))
        raw_inds = r_matrix - tf.reshape(r, [-1, 1])
        clipped_inds = tf.clip_by_value(raw_inds, 0,
                                        distance_biases - 1)
        values = tf.nn.embedding_lookup(distance_bias, clipped_inds)

    return values

def attention_softmax3d(values):
    """
    Performs a softmax over the attention values.

    :param values: 3d tensor with raw values
    :return: 3d tensor, same shape as input
    """
    original_shape = tf.shape(values)
    num_units = original_shape[2]
    reshaped_values = tf.reshape(values, tf.stack([-1, num_units]))
    softmaxes = tf.nn.softmax(reshaped_values)
    return tf.reshape(softmaxes, original_shape)

def computeIntraAttention(sentence, representationSize, numUnits, distance_biases, dropout, reuse_weights=False):
    """
    Compute the intra attention of a sentence. It returns a concatenation
    of the original sentence with its attended output.

    :param sentence: tensor in shape (batch, time_steps, num_units)
    :return: a tensor in shape (batch, time_steps, 2*num_units)
    """
    time_steps = tf.shape(sentence)[1]
    with tf.variable_scope('intra-attention') as scope:
        # this is F_intra in the paper
        # f_intra1 is (batch, time_steps, num_units) and
        # f_intra1_t is (batch, num_units, time_steps)
        f_intra = feedForwardWithRelus(sentence, representationSize, scope, numUnits, dropout=dropout, reuse_weights=reuse_weights)
        f_intra_t = tf.transpose(f_intra, [0, 2, 1])

        # these are f_ij
        # raw_attentions is (batch, time_steps, time_steps)
        raw_attentions = tf.matmul(f_intra, f_intra_t)

        # bias has shape (time_steps, time_steps)
        with tf.device('/cpu:0'):
            bias = get_distance_biases(time_steps, distance_biases, reuse_weights=reuse_weights)

        # bias is broadcast along batches
        raw_attentions += bias
        attentions = attention_softmax3d(raw_attentions)
        attended = tf.matmul(attentions, sentence)

    return tf.concat(axis=2, values=[sentence, attended])