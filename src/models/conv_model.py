import tensorflow as tf

def model():
    """Creates and compiles a Convolutional Neural Network model.

    Returns:
    tf.keras.Model: Compiled CNN model.
    """
    Conv1 = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), input_shape=[56, 193, 1], activation="elu")
    Maxpooling1 = tf.keras.layers.MaxPool2D()
    Conv2 = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation="elu")
    Conv22 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation="elu")
    Maxpooling2 = tf.keras.layers.MaxPool2D()
    Conv3 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation="elu")
    Conv33 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation="elu")
    Maxpooling3 = tf.keras.layers.MaxPool2D()
    flatten = tf.keras.layers.Flatten()
    Dense1 = tf.keras.layers.Dense(128, activation="relu")
    dense2 = tf.keras.layers.Dense(1, activation="sigmoid")
    
    model = tf.keras.Sequential([Conv1, Maxpooling1, Conv2, Conv22, Maxpooling2, Conv3, Conv33, Maxpooling3, flatten, Dense1, dense2])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=[tf.keras.metrics.AUC()])
    
    return model
    
model = model()

from tensorflow.keras.utils import plot_model
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='logs')
    
model.summary()