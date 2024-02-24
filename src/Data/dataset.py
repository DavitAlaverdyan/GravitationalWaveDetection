import math
from random import shuffle
import tensorflow as tf

class Dataset(tf.keras.utils.Sequence):
    """Custom dataset class to handle data batching for training or inference.

    Attributes:
    data (list or numpy.ndarray): Input data.
    y (list or numpy.ndarray, optional): Target labels. Defaults to None.
    batch_size (int, optional): Size of each batch. Defaults to 256.
    shuffle (bool, optional): Flag indicating whether to shuffle data. Defaults to True.
    """
    def __init__(self, data, y=None, batch_size=256, shuffle=True):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        if y is not None:
            self.is_train = True
        else:
            self.is_train = False
            
        self.y = y
        
    def __len__(self):
		
        """Get the number of batches in the dataset.

        Returns:
        int: Number of batches.
        """
        return math.ceil(len(self.data) / self.batch_size)
    
    def __getitem__(self, ids):
        """Get a batch of data and corresponding labels (if available).

        Args:
        ids (int): Index of the batch.

        Returns:
        tuple: Tuple containing batch data and labels (if available).
        """
        batch_data = self.data[ids * self.batch_size : (ids + 1) * self.batch_size]
        
        if self.y is not None:
            batch_y = self.y[ids * self.batch_size : (ids + 1) * self.batch_size]
            
        batch_x = np.array([increase_dimension(x, self.is_train) for x in batch_data])  # Make sure increase_dimension is defined or imported
        batch_x = np.stack(batch_x)
        
        if self.is_train:
            return batch_x, batch_y
        else:
            return batch_x

    def on_epoch_end(self):
        """Function called at the end of each epoch to shuffle the dataset if needed."""
        if self.shuffle and self.is_train:
            ids_y = list(zip(self.data, self.y))
            shuffle(ids_y)
            self.data, self.y = list(zip(*ids_y))