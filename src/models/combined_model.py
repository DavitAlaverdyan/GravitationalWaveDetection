from tensorflow.keras.layers import Input, Model, Reshape
from tensorflow.keras.optimizers import Adam
from .conv_model import Model
from .conv1d_model import Model2


# Input shapes for the three signals
input_layer1 = Input(shape=(1, 4096, 1))
input_layer2 = Input(shape=(1, 4096, 1))
input_layer3 = Input(shape=(1, 4096, 1))

# Create Model2
output_1 = Model2(1, 0.2)(input_layer1, input_layer2, input_layer3)

# Define the model
model2 = Model(
    inputs=[input_layer1, input_layer2, input_layer3], outputs=output_1)

# Compile Model2
model2.compile(
    loss='binary_crossentropy',
    optimizer=Adam(learning_rate=0.003),
    metrics=['accuracy']
)

# Summary of Model2
model2.summary()

# Reshape the output of Model2 to (128, 128, 3)
reshaped_output_1 = Reshape((128, 128, 3))(output_1)

# Instantiate first model
first_model = model()  # Replace `model()` with your actual model creation function

# Pass the reshaped output of Model2 through the first_model
output_2 = first_model(reshaped_output_1)

# Create a combined model
combined_model = Model(
    inputs=[input_layer1, input_layer2, input_layer3],
    outputs=output_2
)

# Compile the combined model
combined_model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(learning_rate=0.003),
    metrics=['accuracy']
)

# Summary of the combined model
combined_model.summary()