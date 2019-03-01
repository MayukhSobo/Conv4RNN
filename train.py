from batch import BatchIterator
import os
import numpy as np
from tqdm import tqdm
from model import Model
import tensorflow as tf

class CNNText(BatchIterator):
    
    def __init__(self, train_path, valid_path, epochs, batch_size, **kwargs):
        X_train_name = kwargs.get('train_name', 'X_train')
        y_train_name = kwargs.get('train_name_y', 'y_train')
        
        X_validation_name = kwargs.get('validation_name', 'X_valid')
        y_validation_name = kwargs.get('validation_name_y', 'y_valid')
        
        fmt = kwargs.get('format', 'npy')
        
        X_train_path = os.path.join(train_path, X_train_name + '.' + fmt)
        y_train_path = os.path.join(train_path, y_train_name + '.' + fmt)
        
        X_validation_path = os.path.join(valid_path, X_validation_name + '.' + fmt)
        y_validation_path = os.path.join(valid_path, y_validation_name + '.' + fmt)
        
        if os.path.exists(X_train_path) and os.path.isfile(X_train_path):
            X_train = np.load(X_train_path)
        else:
            X_train = None

        if os.path.exists(X_validation_path) and os.path.isfile(X_validation_path):
            X_valid = np.load(X_validation_path)
        else:
            X_valid = None
            
        if os.path.exists(y_train_path) and os.path.isfile(y_train_path):
            y_train = np.load(y_train_path)
        else:
            y_train = None
        
        if os.path.exists(y_validation_path) and os.path.isfile(y_validation_path):
            y_valid = np.load(y_validation_path)
        else:
            y_valid = None

        super(self.__class__, self).__init__(X_train, y_train, X_valid, y_valid, epochs, batch_size)
        
        ## 
    
    @staticmethod
    def _train_tf(self, logdir, tolerance):
        pass
    
    def train(self, tolerance, logdir, backend='tensorflow'):
        """
        1. Load the the batch from training data
        2. At every 20th step get the batch from validation data
        3. Pass the validation data for loss calculations
        4. Keep reporting the training and validation loss
        """
        with tf.Graph().as_default(): # In the scope of a tf session with default values
            
            # Create the computation graph for training
            with tf.variable_scope('cnn', reuse=None):
                m = Model(
                    nkernels=100,
                    min_filter=3,
                    max_filter=5,
                    vocab_size=15000,
                    num_class=2,
                    max_len=51,
                    l2_reg=1,
                    device='cpu')

            # Create the computaion graph for validation
            with tf.variable_scope('cnn', reuse=True):
                mtest = Model(
                    nkernels=100,
                    min_filter=3,
                    max_filter=5,
                    vocab_size=15000,
                    num_class=2,
                    max_len=51,
                    l2_reg=1,
                    device='cpu', 
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

                for epoch in range(1, self.epochs+1):
                    all_batches = tqdm(range(self.batches), ascii=True, desc=f'Epoch {epoch}')
                    for i in all_batches:
                        # Load the the batch from training data
                        xs, ys = self.next_train_batch()
                        # Pass the data for fitting
                        
                        # At every 20th step get the batch from validation data

                        # Pass the data for loss calculations

                        # Keep reporting the training and validation loss
                        all_batches.set_postfix({
                            "train_loss": 0.1 * (epoch + i),
                            "valid_loss": 0.1 * (epoch + i)
                        })
















                