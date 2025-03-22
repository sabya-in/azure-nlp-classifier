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

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load Data
df = pd.read_csv("~/dev/AI-sentiment-analysis/azure-nlp-classifier/data/mood_data.csv")

# Preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text

df['processed_text'] = df['sentence'].apply(preprocess_text)

# Encode mood labels
label_encoder = LabelEncoder()
df['mood_encoded'] = label_encoder.fit_transform(df['mood'])

# Tokenization
tokenizer = Tokenizer(num_words=5000)  # Keep only the top 5000 words
tokenizer.fit_on_texts(df['processed_text'])
X_seq = tokenizer.texts_to_sequences(df['processed_text'])

# Padding to ensure all sequences have the same length
X_padded = pad_sequences(X_seq, maxlen=50)

# Convert labels to categorical (for multi-class classification)
y_categorical = to_categorical(df['mood_encoded'])

X_train, X_test, y_train, y_test = train_test_split(X_padded, y_categorical, test_size=0.2, random_state=42)

# Define LSTM Model
model = Sequential([
    Embedding(input_dim=5000, output_dim=128, input_length=50),
    SpatialDropout1D(0.2),
    LSTM(100, dropout=0.2, recurrent_dropout=0.2),
    Dense(y_categorical.shape[1], activation='softmax')  # Softmax for multi-class classification
])

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))

# Save the trained model
model.save("mood_lstm_model.h5")

# Save the tokenizer for preprocessing during inference
with open('tokenizer.json', 'w') as f:
    json.dump(tokenizer.to_json(), f)

# Save the label encoder (optional)
joblib.dump(label_encoder, "label_encoder.pkl")

print("Model, tokenizer, and label encoder saved!")

