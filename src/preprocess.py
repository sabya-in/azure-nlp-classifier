import pandas as pd
import re
import nltk
import joblib
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from azureml.core import Dataset, Workspace

# Download stopwords
nltk.download('stopwords')
stop_words = set(nltk.corpus.stopwords.words('english'))

# Connect to Azure ML workspace
ws = Workspace.from_config()

# Load Data from Azure ML Data Asset
dataset = Dataset.get_by_name(ws, name="mood_data")  # Data Asset Name
df = dataset.to_pandas_dataframe()

# Text Preprocessing Function
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
tokenizer = Tokenizer(num_words=5000)  # Keep only top 5000 words
tokenizer.fit_on_texts(df['processed_text'])
X_seq = tokenizer.texts_to_sequences(df['processed_text'])

# Padding to ensure equal length
X_padded = pad_sequences(X_seq, maxlen=50)

# Convert labels to categorical (for multi-class classification)
y_categorical = pd.get_dummies(df['mood_encoded']).values

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X_padded, y_categorical, test_size=0.2, random_state=42)

# Save preprocessed data
with open("X_train.pkl", "wb") as f: pickle.dump(X_train, f)
with open("X_test.pkl", "wb") as f: pickle.dump(X_test, f)
with open("y_train.pkl", "wb") as f: pickle.dump(y_train, f)
with open("y_test.pkl", "wb") as f: pickle.dump(y_test, f)
joblib.dump(tokenizer, "tokenizer.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

print("Preprocessing complete. Data saved as Pickle files.")

