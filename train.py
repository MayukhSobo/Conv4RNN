from batch import BatchIterator
import os
import numpy as np
from tqdm import tqdm
from model import Model
import tensorflow as tf
import time
from datetime import datetime

class CNNText(BatchIterator):
    
    def __init__(self, config, train_path, valid_path, **kwargs):
        X_train_name = kwargs.get('train_name', 'X_train')
        y_train_name = kwargs.get('train_name_y', 'y_train')
        
        X_validation_name = kwargs.get('validation_name', 'X_valid')
        y_validation_name = kwargs.get('validation_name_y', 'y_valid')
        
        fmt = kwargs.get('fmt', 'npy')
        
        X_train_path = os.path.join(train_path, X_train_name + '.' + fmt)
        y_train_path = os.path.join(train_path, y_train_name + '.' + fmt)
        
        X_validation_path = os.path.join(valid_path, X_validation_name + '.' + fmt)
        y_validation_path = os.path.join(valid_path, y_validation_name + '.' + fmt)
        
#         print(train_path)
#         print(valid_path)
        
        if os.path.exists(X_train_path) and os.path.isfile(X_train_path):
            X_train = np.load(X_train_path)
        else:
            raise ValueError("X_train can not be loaded")
#             X_train = None

        if os.path.exists(X_validation_path) and os.path.isfile(X_validation_path):
            X_valid = np.load(X_validation_path)
        else:
            raise ValueError("X_valid can not be loaded")
#             X_valid = None
            
        if os.path.exists(y_train_path) and os.path.isfile(y_train_path):
            y_train = np.load(y_train_path)
        else:
            raise ValueError("y_train can not be loaded")
#             y_train = None
        
        if os.path.exists(y_validation_path) and os.path.isfile(y_validation_path):
            y_valid = np.load(y_validation_path)
        else:
            raise ValueError("y_valid can not be loaded")
#             y_valid = None
        
        self.nkernels = config['arch']['cnn']['units']
        self.min_filter = config['arch']['cnn']['min_filter']
        self.max_filter = config['arch']['cnn']['max_filter']
        self.l2_reg = config['arch']['cnn']['kernel_l2_reg']
        self.device = config['arch']['fit']['device'].lower()
        self.vocab_size = config['arch']['data']['vocab_size']
        self.embd_size = config['arch']['data']['embedding']
        self.optimizer = config['arch']['fit']['optimizer']
        self.dropout = config['arch']['layers']['dropout']
        
        self.learning_rate = config['arch']['fit']['learning_rate']
        self.backend = config['arch']['fit']['backend']
        epochs = config['arch']['fit']['epochs']
        self.batch_size = config['arch']['fit']['batch_size']
        self.root = config['arch']['data']['root_path']
        self.pretrained_init = config['arch']['initialisation']['embedding']
        
        super(self.__class__, self).__init__(X_train, y_train, X_valid, y_valid, epochs, self.batch_size)
        
        ## 
    
    @staticmethod
    def _train_tf(self, logdir, tolerance):
        pass
    
    def train(self, logdir):
        """
        1. Load the the batch from training data
        2. At every 20th step get the batch from validation data
        3. Pass the validation data for loss calculations
        4. Keep reporting the training and validation loss
        """
        max_steps = self.batches * self.epochs
        
        
        with tf.Graph().as_default(): # In the scope of a tf session with default values
            
            # Create the computation graph for training
            with tf.variable_scope('cnn', reuse=None):
                m = Model(
                    nkernels=self.nkernels,
                    min_filter=self.min_filter,
                    max_filter=self.max_filter,
                    vocab_size=self.vocab_size,
                    num_class=2,
                    max_len=self.max_len,
                    l2_reg=self.l2_reg,
                    esize=self.embd_size,
                    bsize=self.batch_size,
                    optim=self.optimizer,
                    dropout=self.dropout,
                    device=self.device)

            # Create the computaion graph for validation
            with tf.variable_scope('cnn', reuse=True):
                mtest = Model(
                    nkernels=self.nkernels,
                    min_filter=self.min_filter,
                    max_filter=self.max_filter,
                    vocab_size=self.vocab_size,
                    num_class=2,
                    max_len=self.max_len,
                    l2_reg=self.l2_reg,
                    esize=self.embd_size,
                    bsize=self.batch_size,
                    optim=self.optimizer,
                    dropout=self.dropout,
                    device=self.device, 
                    subset='valid')

            # Create a saver to save the graph variables for later
            # https://www.tensorflow.org/api_docs/python/tf/train/Saver
            saver = tf.train.Saver(tf.global_variables())

            # Now declare the path where model checkpoints are to be stored
            save_path = os.path.join(logdir, 'model.ckpt')

            # Object to merge all the summaries to report
            summary = tf.summary.merge_all()
            
            # Create a session with some default config values
            # and `log_device_placement`= False for not to log
            # the device/GPU/CPU info.
            with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
                # Create a summary writter for the session variables
                summary_writer = tf.summary.FileWriter(
                    logdir,
                    graph=sess.graph)

                # Initialize the global variables in computation Graph
                # Whenever we need to perform someting on computaion graph,
                # we need to use session varible and use `run` method in the
                # session variable.
                sess.run(tf.global_variables_initializer())
                
                current_lr = self.learning_rate
#                 lowest_loss_value = float("inf")
                global_step = 0
#                 step_loss_ascend = 0
                
                if self.pretrained_init:
                    print('Initializing the embedding layer with pretrained vectors')
                    init = np.load(os.path.join(self.root, 'rank_matrix.npy'))
                    m.assign_embedding(sess, init)

#                 all_epochs = tqdm(range(1, self.epochs+1), ascii=True)
                def eval_once(mtest, sess):
                    test_loss = 0.0
                    test_accuracy = 0
                    for _ in range(self.batches):
                        x_batch, y_batch = self.next_valid_batch()
                        # x_batch = np.array(x_batch)
                        loss_value, true_count_value = sess.run([mtest.total_loss, mtest.true_count_op], 
                            feed_dict={mtest.inputs: x_batch, mtest.labels: y_batch})
                        test_loss += loss_value
                        test_accuracy += true_count_value
                    test_loss /= self.batches
                    test_accuracy /= (1.0 * self.batches * self.batch_size)
#                     data_loader.reset_pointer()
                    return (test_loss, test_accuracy)
                
                for epoch in range(1, self.epochs+1):
                    train_loss = 0.0
                    true_count_total = 0
                    for i in range(self.batches):
                        # Assign a learning_rate
                        m.assign_lr(sess, current_lr)
                        global_step += 1
                        start_time = time.time()
                        # Load the the batch from training data
                        xs, ys = self.next_train_batch()
                        # Pass the data for fitting
                        # Fitting the training data is rather simple.
                        # To pass values from outside to the computation graph
                        # to fit, we use someting called `feed_dict`. 
                        # `feed_dict` would have variable reference and the value
                        # that we want to pass into that variable.

                        ## Creating the feed dict now.
                        feed = {m.inputs: xs, m.labels: ys}
                        
                        ## Pass the feed_dict into the session and run it.
                        _, loss, tp = sess.run(
                            [m.train_op, m.total_loss, m.true_count_op], 
                            feed_dict=feed)
                        
                        duration = time.time() - start_time
                        
                        train_loss += loss
                        true_count_total += tp
                        
                        if global_step % 200 == 0:
                            summary_str = sess.run(summary)
                            summary_writer.add_summary(summary_str, global_step)
                            
                        if global_step % 10 == 0:
                            examples_per_sec = 50 / duration

                            format_str = ('step %d/%d (epoch %d/%d), loss = %.6f (%.1f examples/sec; %.3f sec/batch), lr: %.6f')
                            print (format_str % (global_step, max_steps, epoch, self.epochs, loss, 
                                examples_per_sec, duration, current_lr))
#                         if loss < lowest_loss_value:
#                             lowest_loss_value = loss
#                             step_loss_ascend = 0
#                         else:
#                             step_loss_ascend += 1
                        
                        
#                         if step_loss_ascend >= 500:
#                             current_lr *= 0.95
                        
#                         if current_lr < 1e-5: break
                
                        # At every 10th step get the batch from validation data
                        
                        # Pass the data for loss calculations

                        # Keep reporting the training and validation loss

                    train_loss /= self.batches
                    train_accuracy = true_count_total * 1.0 / (self.batches * self.batch_size)
                    print("Epoch %d: training_loss = %.6f, training_accuracy = %.3f" % (epoch, train_loss, train_accuracy))
                    test_loss, test_accuracy = eval_once(mtest, sess)
                    print("Epoch %d: test_loss = %.6f, test_accuracy = %.3f" % (epoch, test_loss, test_accuracy))
#                     print(train_loss)
#                     print(train_accuracy)
