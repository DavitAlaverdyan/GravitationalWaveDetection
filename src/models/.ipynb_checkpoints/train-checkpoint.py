# Import necessary libraries
from sklearn.model_selection import train_test_split
from .data.data_processing import Dataset  # Import Dataset class from the appropriate location
from models.model import create_model  # Import model creation function from the models folder
import pandas as pd

# Load and preprocess the data

train_labels = pd.read_csv("G:/W_project/Waves_data/g2net-gravitational-wave-detection/training_labels.csv")  
# Load training labels here

sample_submission = pd.read_csv("G:/W_project/Waves_data/g2net-gravitational-wave-detection/submission_labels.csv")
# Load sample submission data here

# Get indices and target labels from the training and test data
train_idx = train_labels['id'].values
y = train_labels['target'].values
y1 = sample_submission['target'].values
test_idx = sample_submission['id'].values

# Split the data into training and validation sets
x_train, x_valid, y_train, y_valid = train_test_split(train_idx, y, test_size=0.05, random_state=42, stratify=y)

# Create Dataset instances for training, validation, and test data
train_dataset = Dataset(x_train, y_train)  # Initialize Dataset for training data
valid_dataset = Dataset(x_valid, y_valid)  # Initialize Dataset for validation data
test_dataset = Dataset(test_idx, y1)       # Initialize Dataset for test data

# Create and compile the model
model = create_model()  # Implement create_model function in models/model.py file

# Train the model
model.fit(train_dataset, epochs=2, validation_data=valid_dataset)

# Predict on the test dataset
preds = model.predict(test_dataset)
preds = preds.reshape(-1)

# Create a DataFrame for submission
submission = pd.DataFrame({'id':sample_submission['id'],'target':preds})
submission.to_csv('submission.csv',index=False)

