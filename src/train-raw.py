import argparse
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib

# Argument parsing (Azure ML passes dataset path as an argument)
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, help="Path to input CSV file")
parser.add_argument("--output_dir", type=str, help="Path to save model")
args = parser.parse_args()

# Load data
print(f"Loading data from {args.data_path}")
df = pd.read_csv(args.data_path)

# Ensure correct column names (modify these based on your actual file structure)
TEXT_COLUMN = "sentence"
LABEL_COLUMN = "mood"

if TEXT_COLUMN not in df.columns or LABEL_COLUMN not in df.columns:
    raise ValueError(f"Columns {TEXT_COLUMN} and {LABEL_COLUMN} must exist in the dataset")

# Preprocessing
X_train, X_test, y_train, y_test = train_test_split(df[TEXT_COLUMN], df[LABEL_COLUMN], test_size=0.2, random_state=42)

# Define a text classification pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", MultinomialNB())
])

# Train the model
print("Training model...")
model.fit(X_train, y_train)

# Evaluate the model
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy:.2f}")

# Save the model
os.makedirs(args.output_dir, exist_ok=True)
model_path = os.path.join(args.output_dir, "mood_model.pkl")
joblib.dump(model, model_path)
print(f"Model saved at {model_path}")

