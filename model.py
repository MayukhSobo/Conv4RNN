import tensorflow as tf


class Model:

    def __init__(self, nkernels, min_filter, max_filter,
                 vocab_size, num_class, max_len, l2_reg,
                 subset='train', esize=300, bsize=50, **kwargs):

        self.is_train = subset == 'train'
        self.emb_size = esize
        self.batch_size = bsize
        self.num_kernel = nkernels
        self.min_filter = min_filter
        self.max_filter = max_filter
        self.vocab_size = vocab_size
        self.num_class = num_class
        self.sent_len = max_len
        self.l2_reg = l2_reg
        self.optimizer = None
        self.dropout_rate = 0

        if self.is_train:
            self.optimizer = kwargs.get('optim', 'adam')
            self.dropout_rate = kwargs.get('dropout', 0.5)

        self._build_graph()

    def _build_graph(self):
        """ Build the computation graph.
        Step 1: Create I/O placeholders
        Step 2: Build an embddeding layer
        Step 3: Add Conv units with maxpool layers
        Step 4: Add Dropout if required
        Step 5: Add Fully connected layers
        Step 6: Configure Loss and learning rate with optimizers

        """

        ### ------------------ Create the I/O placeholders ------------------------ ##
        # Input contains a matrix of batch_size X max_sent_len
        # This is a vector or padded rank vectors for each review for each data points
        self._input = tf.placeholder(dtype=tf.int64, shape=[self.batch_size, self.sent_len], name='input_x')
        # Output should contain predictions for each batch inputs, not one hot encoded yet.
        self._labels = tf.placeholder(dtype=tf.int64, shape=[self.batch_size], name='input_y')
        # This is a placeholder for regularisation losses for each layers in convNet.
        regularization_loss = []

        ### ------------------ Build an embddeding layer ------------------------ ##

        with tf.variable_scope('embeddings'):
            ## Embedding is required because we have ranks of each words for the convolution
            ## and with ranks we need to represent each rank(word) into it's trainable
            ## weight vectors. We can initialize these with random uniform values and then train
            ## these weights during training or we can initialize these with pretrained
            ## word vector and then train these. We shall parametrized this decision.
            ## The reason to convert this single ranks to weight vector is to facilitate
            ## learning sematics between the words. With single value it would be difficult.
            ## In case of LSTMs also, we perform this embeddings.
            self._Wemb = Model._variable_on_cpu(
                name='embed',
                shape=[self.vocab_size, self.emb_size],
                initializer=tf.random_uniform_initializer(minval=-1.0, maxval=1.0)
            )
            ## This is basically looking/performing the actual embeddings on batch data
            batch_emb = tf.nn.embedding_lookup(params=self._Wemb, ids=self._input)
            batch_emb = tf.expand_dims(batch_emb, -1)  # Because it need to be 4-dimensional as CNN requires 4d inputs
            # These 4-d inputs are N x H x W x C. The actual input was only a rank/id vector each of size `self.sent_len`
            # and the dataset would have been N x self.sent_len. Embedding and expand dims are to get 4-d data. `batch_emb`
            # hence would go directly into the conv layer.

        ### ------------------ Add Conv units with maxpool layers ------------------------ ##

        with tf.variable_scope('conv'):
            conv_layers = []  # Here we shall store all the conv layers

            # As per the research paper https://arxiv.org/pdf/1408.5882.pdf section 3.1
            # They are using 100 conv units or feature maps with 3 layers with kernel sizes
            # 3, 4 and 5 with kernel sizes. Which means there would 3 layers of convolution and maxpool
            for ks in range(self.min_filter, self.max_filter + 1):
                # Initialize the kernel weights and reg_loss
                kernel, reg_loss = Model._variable_with_weight_decay(
                    name=f'kernel_{ks}',
                    # Kernel size is always 4-d like input.
                    shape=[ks, self.emb_size, 1, self.num_kernel],
                    initializer=tf.truncated_normal_initializer(stddev=0.01),
                    reg_alpha=self.l2_reg)
                regularization_loss.append(reg_loss)

                # Create the conv layers wiht kernel weights
                # We are using strides of 1x1 and valid padding
                conv = tf.nn.conv2d(input=batch_emb, filter=kernel, strides=[1, 1, 1, 1], padding='VALID')

                # Create the bias for each conv units. There were 100 feature maps or conv units
                # hence we would have 100 bias terms.
                bias = Model._variable_on_cpu(
                    name=f'bias_{ks}',
                    # `self.num_kernel` = 100
                    shape=[self.num_kernel],
                    initializer=tf.constant_initializer(0.0))

                # Now add all the conv units and their bias values
                c = tf.nn.bias_add(conv, bias)

                # Now apply activation function on all conv units, Polular choice is ReLu.
                # This is the output from one conv layer
                c_activated = tf.nn.relu(c, name='activation')

                # At the end of ReLu, the shape of the output would be [batch_size x conv_len x 1 x conv_units]
                # In this case it would be [50 x ? x 1 x 100]
                # conv_len -> After applyting the filter with strides 1x1 and padding on input

                conv_len = c_activated.get_shape()[1]

                # Now we are applying max pooling
                # This maxpool would choose among all `conv_len`
                # values being represented by 100 conv units and the
                # would choose one value from each `conv_len` which
                # means previously we were having `conv_len` outputs from
                # each conv units and hence dimension were `conv_len` x 100.
                # But now it would choose only a single values from each 100 convnet
                # and hence it would have 100 values 1 from each conv unit. Hence the
                # size of max pool should be [1 x conv_len x 1 x 1] because it would
                # perform pooling only on `conv_len` and the output shape would be N x 1 x 1 x 100
                # We shall then convert it into 2d from 4d of dimension N x 100
                c_activated_max_pooled = tf.nn.max_pool(
                    c_activated,
                    ksize=[1, conv_len, 1, 1],
                    strides=[1, 1, 1, 1], padding='VALID')
                # Convert into 2d as mentioned above
                # Squeezing the 1st and 2nd dim and keeping 0th and 3rd dim
                c_activated_max_pooled_2d = tf.squeeze(c_activated_max_pooled, axis=[1, 2])

                # Now strore the Conv-maxpool layer
                conv_layers.append(c_activated_max_pooled_2d)

            # Now merge all the conv layers..We may not need it if we can interlink
            # all the conv layers manually. But this seems more cleaner
            c_all = tf.concat(values=conv_layers, axis=1, name='pool')

        ### ------------------ Add Dropout if required ------------------------ ##
        # As per the research paper, they use a dropout rate of 0.5... But we shall add
        # dropout only during training
        if self.is_train:
            d = tf.nn.dropout(c_all, 1 - self.dropout_rate)
        else:
            d = c_all

        ### ------------------ Add Fully connected layers ------------------------ ##
        # The size of a fully connected layer has a formula governed by
        # (last_conv_layer_kernel_size - initial_conv_layer_kernel_size + 1) * conv_units_each_layer
        # Hence in our case it would be (5 - 3 + 1) * 100 = 300
        # It's a sigmoid layer
        fc_size = (self.max_filter - self.min_filter + 1) * self.num_kernel
        with tf.variable_scope('dense'):
            fc, fc_reg_loss = Model._variable_with_weight_decay(
                name='dense',
                shape=[fc_size, self.num_class],
                initializer=tf.truncated_normal_initializer(stddev=0.05), reg_alpha=self.l2_reg)
            regularization_loss.append(fc_reg_loss)

            ## Add bias
            bias = Model._variable_on_cpu(
                'fc_bias',
                [self.num_class],
                tf.constant_initializer(0.01))

            output = tf.nn.bias_add(tf.matmul(d, fc), bias)

        ### ------------------ Configure Loss and learning rate with optimizers ------------------------ ##
        # This SO post https://stackoverflow.com/questions/35241251/in-tensorflow-what-is-the-difference-between-sampled-softmax-loss-and-softmax-c
        # it makes sense not to use `sampled_softmax_loss` in fovour of `softmax_cross_entropy_with_logits` because
        # we have only 2 class labels
        # Also we are not using One-hot-encoded version of the class label because our self._labels is of size (N, ) and type int32
        # More detail here https://stackoverflow.com/questions/37312421/whats-the-difference-between-sparse-softmax-cross-entropy-with-logits-and-softm
        cross_entropy_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self._labels,
                logits=output, name='softmax'),
            name='output_loss')
        regularization_loss.append(cross_entropy_loss)
        self._total_loss = tf.add_n(regularization_loss, name='total_loss')

        ### ------------------ Calculate number of correct predictions ------------------------ ##

        correct_prediction = tf.to_int32(tf.nn.in_top_k(output, self._labels, 1))
        self._true_predictions = tf.reduce_sum(correct_prediction)

        pass

    @staticmethod
    def _variable_with_weight_decay(name, shape, initializer, reg_alpha):
        """
        It returns two variables.
        1. A simple weight vector on CPU.
        2. A variable which is used for regularization parameter
        """
        w = Model._variable_on_cpu(name, shape, initializer)
        # If we want to use any regularisation for kernel weights
        if reg_alpha > 0.0:
            # This is l2 regularisation with hyperparam reg_alpha.
            reg_loss = tf.multiply(tf.nn.l2_loss(w), reg_alpha, name='reg_loss')
        # If we don't want to regularize the the CNN kernel weights
        else:
            reg_loss = tf.constant(0.0, dtype=tf.float32)

        return w, reg_loss

    @staticmethod
    def _variable_on_cpu(name, shape, initializer):
        """
        Declare a variable on CPU of `shape` and initialize
        those with the `initializer`.
        """
        with tf.device('/cpu:0'):
            var = tf.get_variable(name, shape, initializer=initializer)
        return var


if __name__ == '__main__':
    m = Model(
        nkernels=100,
        min_filter=3,
        max_filter=5,
        vocab_size=15000,
        num_class=2,
        max_len=51,
        l2_reg=1,
    )
    print(m)