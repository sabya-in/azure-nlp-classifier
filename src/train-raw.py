import pandas as pd
import re
import nltk
import joblib
import json
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from azureml.core import Run, Dataset

# Azure ML run context
run = Run.get_context()

# Load Dataset
ws = run.experiment.workspace
dataset = Dataset.get_by_name(ws, name='mood_data')
df = dataset.to_pandas_dataframe()

# Download stopwords
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

# Preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text
    
# Check if 'sentence' and 'mood' header exists
if 'sentence' not in df.columns or 'mood' not in df.columns:
    raise ValueError("CSV file must contain 'sentence' and 'mood' columns.")

df['processed_text'] = df['sentence'].apply(preprocess_text)

# Encode mood labels
label_encoder = LabelEncoder()
df['mood_encoded'] = label_encoder.fit_transform(df['mood'])

# Tokenization
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(df['processed_text'])
X_seq = tokenizer.texts_to_sequences(df['processed_text'])

# Padding sequences
X_padded = pad_sequences(X_seq, maxlen=50)

# Convert labels to categorical
y_categorical = to_categorical(df['mood_encoded'])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_padded, y_categorical, test_size=0.2, random_state=42)

# Define LSTM Model
model = Sequential([
    Embedding(input_dim=5000, output_dim=128, input_length=50),
    SpatialDropout1D(0.2),
    LSTM(100, dropout=0.2, recurrent_dropout=0.2),
    Dense(y_categorical.shape[1], activation='softmax')
])

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))

# Save Model and Preprocessing Files
model.save("mood_lstm_model.h5")

with open('tokenizer.json', 'w') as f:
    json.dump(json.loads(tokenizer.to_json()), f)

joblib.dump(label_encoder, "label_encoder.pkl")

# Log Model & Metrics in Azure ML
run.log("accuracy", model.evaluate(X_test, y_test)[1])
run.upload_file("mood_lstm_model.h5", "outputs/mood_lstm_model.h5")
run.upload_file("tokenizer.json", "outputs/tokenizer.json")
run.upload_file("label_encoder.pkl", "outputs/label_encoder.pkl")

print("Model and preprocessing files saved to Azure ML.")

