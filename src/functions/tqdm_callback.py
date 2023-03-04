from tqdm import tqdm
from keras.callbacks import Callback

class TqdmCallback(Callback):
    def __init__(self, epochs):
        self.epochs = epochs
        
    def on_train_begin(self, logs=None):
        self.tqdm_bar = tqdm(total=self.epochs, desc='Training')
        
    def on_epoch_end(self, epoch, logs=None):
        self.tqdm_bar.update(1)
        
    def on_train_end(self, logs=None):
        self.tqdm_bar.close()