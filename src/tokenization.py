import json
import joblib
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# Tokenization
tokenizer = Tokenizer(num_words=4000)  # Keep only the top 5000 words
tokenizer.fit_on_texts(df['processed_text'])
X_seq = tokenizer.texts_to_sequences(df['processed_text'])

# Padding to ensure all sequences have the same length
X_padded = pad_sequences(X_seq, maxlen=100)

# Convert labels to categorical (for multi-class classification)
from tensorflow.keras.utils import to_categorical
Y_categorical = to_categorical(df['mood_encoded'])

# split data for training
X_train, X_test, y_train, y_test = train_test_split(X_padded, Y_categorical, test_size=0.1, random_state=42)

# Save the tokenizer for preprocessing during inference
with open('tokenizer.json', 'w') as f:
    json.dump(tokenizer.to_json(), f)
    
# Save the label encoder
joblib.dump(label_encoder, "label_encoder.pkl")
