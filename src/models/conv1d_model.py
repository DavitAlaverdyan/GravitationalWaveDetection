from tensorflow.keras.layers import Layer, Conv1D, MaxPooling1D, Dropout, BatchNormalization, Reshape, concatenate

class Model2(Layer):

    def __init__(self, filters, rate, pooling=True):
		 """
        Custom Keras layer implementing a sequence of Conv1D and MaxPooling1D operations.

        Args:
        - filters (int): Number of filters for convolutional layers.
        - rate (float): Dropout rate for dropout layers.
        - pooling (bool): Flag to enable or disable pooling layers.

        Returns:
        - A concatenated tensor of processed inputs.
        """
        super(Model2, self).__init__()
        self.pooling = pooling
        self.filters = filters
        self.rate = rate
        self.pool2D = MaxPool2D(pool_size=(2,2))
        self.pool1d = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding='valid')
        
        self.c11 = Conv1D(filters=8, kernel_size=5, strides=1, padding='same', activation='relu')  
        self.c12 = Conv1D(filters=16, kernel_size=5, strides=1, padding='same', activation='relu')  
        self.c13 = Conv1D(filters=32, kernel_size=5, strides=1, padding='same', activation='relu')
        self.c14 = Conv1D(filters=64, kernel_size=5, strides=1, padding='same', activation='relu')
        self.c15 = Conv1D(filters=128, kernel_size=5, strides=1, padding='same', activation='relu')  
        self.drop1 = Dropout(self.rate)
        self.drop2 = Dropout(self.rate)
        self.drop3 = Dropout(self.rate)
        self.drop4 = Dropout(self.rate)
        self.drop5 = Dropout(self.rate)
        self.bn1 = tf.keras.layers.BatchNormalization(momentum=0.99,epsilon=0.001)
        self.bn2 = tf.keras.layers.BatchNormalization(momentum=0.99,epsilon=0.001)
        self.bn3 = tf.keras.layers.BatchNormalization(momentum=0.99,epsilon=0.001)
        self.bn4 = tf.keras.layers.BatchNormalization(momentum=0.99,epsilon=0.001)
        self.bn5 = tf.keras.layers.BatchNormalization(momentum=0.99,epsilon=0.001)
        

    def call(self, x,y,z):
        ##
        # x1 = tf.reshape(x, (-1, 4096, 1))
		        """
        Process the input tensors x, y, and z through convolutional layers and return a concatenated tensor.

        Args:
        - x (tf.Tensor): Input tensor x.
        - y (tf.Tensor): Input tensor y.
        - z (tf.Tensor): Input tensor z.

        Returns:
        - A concatenated tensor of processed inputs.
        """
        print(x)
        x1 = self.bn1(self.drop1(self.c11(x)))
        x1 = tf.keras.layers.Reshape((-1, 8))(x1)
        print(x1)
        x2 = self.pool1d(x1)
        
        x3 = self.bn2(self.drop2(self.c12(x2)))
        x4 = self.pool1d(x3)
                
        x5 = self.bn3(self.drop3(self.c13(x4)))
        x6 = self.pool1d(x5)
        
        x7 = self.bn4(self.drop1(self.c14(x6)))
        x8 = self.pool1d(x7)

        x9 = self.bn5(self.drop5(self.c15(x8)))
        x10 = self.pool1d(x9)
        x10 = tf.keras.layers.Reshape((1, 128, 128))(x10)

        ##
        y1 = self.bn1(self.drop1(self.c11(y)))
        y1 = tf.keras.layers.Reshape((-1, 8))(y1)
        y2 = self.pool1d(y1)

        y3 = self.bn2(self.drop2(self.c12(y2)))
        y4 = self.pool1d(y3)
                
        y5 = self.bn3(self.drop3(self.c13(y4)))
        y6 = self.pool1d(y5)
        
        y7 = self.bn4(self.drop1(self.c14(y6)))
        y8 = self.pool1d(y7)
        
        y9 = self.bn5(self.drop5(self.c15(y8)))
        y10 = self.pool1d(y9)
        y10 = tf.keras.layers.Reshape((1, 128, 128))(y10)
        

        ##
        z1 = self.bn1(self.drop1(self.c11(z)))
        z1 = tf.keras.layers.Reshape((-1, 8))(z1)
        
        z2 = self.pool1d(z1)

        z3 = self.bn2(self.drop2(self.c12(z2)))
        z4 = self.pool1d(z3)
                
        z5 = self.bn3(self.drop3(self.c13(z4)))
        z6 = self.pool1d(z5)
        
        z7 = self.bn4(self.drop1(self.c14(z6)))
        z8 = self.pool1d(z7)

        z9 = self.bn5(self.drop5(self.c15(z8)))
        z10 = self.pool1d(z9)
        z10 = tf.keras.layers.Reshape((1, 128, 128))(z10)
        
        X = concatenate([x10,y10,z10],axis=1)
        
        return X