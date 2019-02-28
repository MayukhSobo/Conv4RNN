from abc import ABC, abstractmethod
import numpy as np

class BatchIterator(ABC):

    train_pointer = 0
    validation_pointer = 0

    def __init__(self, X_train, y_train, X_valid, y_valid, epochs, batch_size):
        self.X_train = X_train
        self.X_valid = X_valid
        
        self.y_train = y_train
        self.y_valid = y_valid

        self.N_train = self.X_train.shape[0]
        self.N_valid = self.X_valid.shape[0]

        self.epochs = epochs
        self.batch_size = batch_size
        self.batches = self.N_train // self.batch_size


    def next_train_batch(self):
        x_out = []
        y_out = []
        for i in range(self.batch_size):
            x_out.append(self.X_train[(BatchIterator.train_pointer + i) % self.N_train])
            y_out.append(self.y_train[(BatchIterator.train_pointer + i) % self.N_train])
        BatchIterator.train_pointer += self.batch_size
        return np.array(x_out), np.array(y_out)
    
    def next_valid_batch(self):
        x_out = []
        y_out = []
        for i in range(self.batch_size):
            x_out.append(self.X_valid[(BatchIterator.validation_pointer + i) % self.N_valid])
            y_out.append(self.y_valid[(BatchIterator.validation_pointer + i) % self.N_valid])
        BatchIterator.validation_pointer += self.batch_size
        return np.array(x_out), np.array(y_out)
    
    @abstractmethod
    def train(self, device='cuda'):
        pass