import argparse
import os
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from azureml.core import Workspace, Dataset, Run
from azureml.core.model import Model

# Parse Arguments (Azure ML Passes Data Path as Argument)
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, help="Path to training data CSV")
parser.add_argument("--output_dir", type=str, help="Path to save the model")
args = parser.parse_args()

# Connect to Azure ML Run Context
run = Run.get_context()
ws = run.experiment.workspace

# Load Data
print(f"Loading data from {args.data_path}")
df = pd.read_csv(args.data_path)  # Assumes columns ['text', 'label']

# Preprocess Text Data
MAX_WORDS = 10000   # Vocabulary size
MAX_LENGTH = 20     # Max sentence length

tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(df["text"])

X = tokenizer.texts_to_sequences(df["text"])
X = pad_sequences(X, maxlen=MAX_LENGTH, padding='post')
y = pd.get_dummies(df["label"]).values  # One-hot encoding for 6 categories

# Build LSTM Model
model = Sequential([
    Embedding(MAX_WORDS, 128, input_length=MAX_LENGTH),
    LSTM(64, return_sequences=True),
    LSTM(32),
    Dropout(0.5),
    Dense(6, activation='softmax')  # 6 categories
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# Train the Model
model.fit(X, y, epochs=5, batch_size=32, validation_split=0.2)

# Save the Model
os.makedirs(args.output_dir, exist_ok=True)
model_path = os.path.join(args.output_dir, "mood_lstm.h5")
model.save(model_path)
print(f"Model saved at {model_path}")

# Log Model to Azure ML
run.log("accuracy", float(model.evaluate(X, y)[1]))
run.upload_file(name="mood_lstm.h5", path_or_stream=model_path)
run.complete()

