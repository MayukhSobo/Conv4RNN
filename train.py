from batch import BatchIterator
import os
import numpy as np
from tqdm import tqdm

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
    def _train_tf(self):
        pass
    
    
    def train(self, backend='tensorflow'):
        for epoch in range(1, self.epochs+1):
            all_batches = tqdm(range(self.batches), ascii=True, desc=f'Epoch {epoch}')
            for i in all_batches:
                xs, ys = self.next_train_batch()
                # Here do the fitting of training data
                
                # Then calculate the train and validation loss
                all_batches.set_postfix({
                    "train_loss": 0.1 * (epoch + i),
                    "valid_loss": 0.1 * (epoch + i)
                })
                