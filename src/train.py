import argparse
import tensorflow as tf
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from azureml.core import Workspace, Dataset, Run
from azureml.core.model import Model

run = Run.get_context()
ws = run.experiment.workspace  # Automatically gets the current workspace
datastore = ws.get_default_datastore()
print(f"Connected to workspace: {ws.name} with datastore: {datastore}")

# Argument parsing (Azure ML passes dataset path as an argument)
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, help="Path to training data CSV")
parser.add_argument("--output_dir", type=str, help="Path to save the model")
args = parser.parse_args()

# Load data
print(f"Loading data from {args.data_path}")
df = pd.read_csv(args.data_path)

# Ensure correct column names (modify these based on your actual file structure)
TEXT_COLUMN = "sentence"
LABEL_COLUMN = "emotion"

if TEXT_COLUMN not in df.columns or LABEL_COLUMN not in df.columns:
    raise ValueError(f"Columns {TEXT_COLUMN} and {LABEL_COLUMN} must exist in the dataset")

# Preprocessing
MAX_WORDS = 10000   # Vocabulary size
MAX_LENGTH = 20     # Max sentence length

tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(df[f"{TEXT_COLUMN}"])

X = tokenizer.texts_to_sequences(df[f"{TEXT_COLUMN}"])
X = pad_sequences(X, maxlen=MAX_LENGTH, padding='post')
y = pd.get_dummies(df[f"{LABEL_COLUMN}"]).values  # One-hot encoding for 6 categories

# Define a model using lstm
model = Sequential([
    Embedding(MAX_WORDS, 128, input_length=MAX_LENGTH),
    LSTM(64, return_sequences=True),
    LSTM(32),
    Dropout(0.5),
    Dense(6, activation='softmax')  # 6 categories
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# Train the model
print("Training model...")
model.fit(X, y, epochs=5, batch_size=32, validation_split=0.2)

# Evaluate the model
accuracy = model.evaluate(X, y)[1]
print(f"Model Accuracy: {accuracy:.2f}")

# Save the model
model_filename = "mood_lstm_model.h5"
model.save(model_filename)
print(f"Model saved as {model_filename}")

datastore.upload_files(
    files=[model_filename],  # List of files to upload
    target_path="models/",   # Path inside the datastore
    overwrite=True
)
print("Model uploaded to datastore successfully!")

model = Model.register(
    workspace=ws,
    model_name="mood_lstm_model",  # Name for Azure ML Model Registry
    model_path="mood_lstm_model.h5",  # Path in datastore
    description="LSTM model for mood classification"
)
print(f"Model registered: {model.name} (version {model.version})")

