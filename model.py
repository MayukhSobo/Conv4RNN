import tensorflow as tf

def _variable_on_device(name, shape, initializer, device='cuda'):
        """
        Declare a variable on CPU of `shape` and initialize
        those with the `initializer`.
        """
        if device == 'cuda':
            d = '/gpu:0'
        else:
            d = '/cpu:0'
        with tf.device(d):
            var = tf.get_variable(name, shape, initializer=initializer)
        return var

def _variable_with_weight_decay(name, shape, initializer, reg_alpha, device):
        """
        It returns two variables.
        1. A simple weight vector on CPU.
        2. A variable which is used for regularization parameter
        """
        w = _variable_on_device(name, shape, initializer, device)
        if device == 'cuda':
            d = '/gpu:0'
        else:
            d = '/cpu:0'
        with tf.device(d):
            # If we want to use any regularisation for kernel weights
            if reg_alpha > 0.0:
                # This is l2 regularisation with hyperparam reg_alpha.
                reg_loss = tf.multiply(tf.nn.l2_loss(w), reg_alpha, name='weight_loss')
            # If we don't want to regularize the the CNN kernel weights
            else:
                reg_loss = tf.constant(0.0, dtype=tf.float32)

        return w, reg_loss

class Model:
    """
    This defines a tensorFLow computaion graph mostly
    keeping the details intact in the research paper.
    
    Conv Max pool 1
    ---------  ---------
    |       |  |       |
    | conv  |__|  max  | 
    | 3 x 3 |  |  pool |
    |  100  |  |       |
    ---------  ---------
    
    Conv Max pool 2
    ---------  ---------
    |       |  |       |
    | conv  |__|  max  | 
    | 4 x 4 |  |  pool |
    |  100  |  |       |
    ---------  ---------  

     Conv Max pool 3
    ---------  ---------
    |       |  |       |
    | conv  |__|  max  | 
    | 5 x 5 |  |  pool |
    |  100  |  |       |
    ---------  ---------  
    """
    def __init__(self, nkernels, min_filter, max_filter,
                 vocab_size, num_class, max_len, l2_reg,
                 esize, bsize, optim, dropout, device, subset='train'):

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
            self.optimizer = optim
            self.dropout_rate = dropout
        
        self.device = device

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
        self._inputs = tf.placeholder(dtype=tf.int64, shape=[self.batch_size, self.sent_len], name='input_x')
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
            self._Wemb = _variable_on_device(
                name='embedding',
                shape=[self.vocab_size, self.emb_size],
                initializer=tf.random_uniform_initializer(minval=-1.0, maxval=1.0),
                device=self.device)
            ## This is basically looking/performing the actual embeddings on batch data
            batch_emb = tf.nn.embedding_lookup(params=self._Wemb, ids=self._inputs)
            # Because it need to be 4-dimensional as CNN requires 4d inputs
            batch_emb = tf.expand_dims(batch_emb, -1) 
            # These 4-d inputs are N x H x W x C. 
            # The actual input was only a rank/id vector each of size `self.sent_len`
            # and the dataset would have been N x self.sent_len. 
            # Embedding and expand dims are to get 4-d data. `batch_emb`
            # hence would go directly into the conv layer.

        ### ------------------ Add Conv units with maxpool layers ------------------------ ##

        with tf.variable_scope('conv'):
            conv_layers = []  # Here we shall store all the conv layers

            # As per the research paper https://arxiv.org/pdf/1408.5882.pdf section 3.1
            # They are using 100 conv units or feature maps with 3 layers with kernel sizes
            # 3, 4 and 5 with kernel sizes. Which means there would 3 layers of convolution and maxpool
            for ks in range(self.min_filter, self.max_filter + 1):
                # Initialize the kernel weights and reg_loss
                kernel, reg_loss = _variable_with_weight_decay(
                    name=f'kernel_{ks}',
                    # Kernel size is always 4-d like input.
                    shape=[ks, self.emb_size, 1, self.num_kernel],
                    initializer=tf.truncated_normal_initializer(stddev=0.01),
                    reg_alpha=self.l2_reg, device=self.device)
                regularization_loss.append(reg_loss)

                # Create the conv layers wiht kernel weights
                # We are using strides of 1x1 and valid padding
                conv = tf.nn.conv2d(input=batch_emb, filter=kernel, strides=[1, 1, 1, 1], padding='VALID')

                # Create the bias for each conv units. There were 100 feature maps or conv units
                # hence we would have 100 bias terms.
                bias = _variable_on_device(
                    name=f'bias_{ks}',
                    # `self.num_kernel` = 100
                    shape=[self.num_kernel],
                    initializer=tf.constant_initializer(0.0), device=self.device)

                # Now add all the conv units and their bias values
                c = tf.nn.bias_add(conv, bias)

                # Now apply activation function on all conv units, Polular choice is ReLu.
                # This is the output from one conv layer
                c_activated = tf.nn.relu(c, name='activation')

                # At the end of ReLu, the shape of the 
                # output would be [batch_size x conv_len x 1 x conv_units]
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
                c_activated_max_pooled_2d = tf.squeeze(c_activated_max_pooled, squeeze_dims=[1, 2])

                # Now strore the Conv-maxpool layer
                conv_layers.append(c_activated_max_pooled_2d)

            # Now merge all the conv layers..We may not need it if we can interlink
            # all the conv layers manually. But this seems more cleaner
            c_all = tf.concat(values=conv_layers, axis=1, name='pool')

        ### ------------------ Add Dropout if required ------------------------ ##
        # As per the research paper, they use a dropout rate of 0.5... But we shall add
        # dropout only during training
        if self.is_train and self.dropout_rate > 0:
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
            fc, fc_reg_loss = _variable_with_weight_decay(
                name='fc',
                shape=[fc_size, self.num_class],
                initializer=tf.truncated_normal_initializer(stddev=0.05), 
                reg_alpha=self.l2_reg,
                device=self.device)
            regularization_loss.append(fc_reg_loss)

            ## Add bias
            bias_fc = _variable_on_device(
                'fc_bias',
                [self.num_class],
                tf.constant_initializer(0.01), self.device)

            output = tf.nn.bias_add(tf.matmul(d, fc), bias_fc)

        ### ------------------ Configure Loss and learning rate with optimizers ------------------------ ##
        # This SO post https://bit.ly/2TmR3Vc
        # it makes sense not to use `sampled_softmax_loss` in 
        # fovour of `softmax_cross_entropy_with_logits` because
        # we have only 2 class labels
        # Also we are not using One-hot-encoded version of the class label 
        # because our self._labels is of size (N, ) and type int32
        # More detail here https://bit.ly/2BZ4wsr
        cross_entropy_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self._labels,
                logits=output, name='cross_entropy_per_example'),
            name='cross_entropy_loss')
        regularization_loss.append(cross_entropy_loss)
        self._total_loss = tf.add_n(regularization_loss, name='total_loss')

        ### ------------------ Calculate number of correct predictions ------------------------ ##

        correct_prediction = tf.to_int32(tf.nn.in_top_k(output, self._labels, 1))
        self._true_count_op = tf.reduce_sum(correct_prediction)

        ### ------------------ Tune learning rate and Optimization algorithm ------------------------ ##

        # This stage would only run during the training phase
        self._learning_rate = tf.Variable(0.0, trainable=False)
        if self.is_train:
            if self.optimizer == 'adadelta':
                algo = tf.train.AdadeltaOptimizer(self._learning_rate)
            elif self.optimizer == 'adagrad':
                algo = tf.train.AdagradOptimizer(self._learning_rate)
            elif self.optimizer == 'adam':
                algo = tf.train.AdamOptimizer(self._learning_rate)
            else:
                raise NotImplementedError(f'Algo {algo} is not implemented yet')

            # This calculates the gradient and applies then with the previous
            # gradient values. If we need need to avoid exploding gradient,
            # we may need to split this section and manually apply the clipping
            # the compelte thing is mentioned https://www.tensorflow.org/api_docs/python/tf/train/Optimizer
            self._train_op = algo.minimize(self._total_loss)
            for var in tf.trainable_variables():
                tf.summary.histogram(var.op.name, var)
        else:
            self._train_op = tf.no_op()
        return True

    @property
    def train_op(self):
        return self._train_op
    
    @property
    def total_loss(self):
        return self._total_loss
    
    @property
    def inputs(self):
        return self._inputs
    
    @property
    def labels(self):
        return self._labels
    
    @property
    def learning_rate(self):
        return self._learning_rate

    @property
    def Wemb(self):
        return self._Wemb

    @property
    def true_count_op(self):
        return self._true_count_op
            

    def assign_lr(self, session, lr_value):
        session.run(tf.assign(self._learning_rate, lr_value))
    
    def assign_embedding(self, session, pretrained):
        session.run(tf.assign(self.Wemb, pretrained))

if __name__ == '__main__':
    m = Model(
        nkernels=100,
        min_filter=3,
        max_filter=5,
        vocab_size=15000,
        num_class=2,
        max_len=51,
        l2_reg=1,
        device='cpu'
    )
    print(m.train_op)