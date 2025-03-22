import tensorflow as tf
import joblib
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D

# Load Preprocessed Data
with open("X_train.pkl", "rb") as f: X_train = pickle.load(f)
with open("X_test.pkl", "rb") as f: X_test = pickle.load(f)
with open("y_train.pkl", "rb") as f: y_train = pickle.load(f)
with open("y_test.pkl", "rb") as f: y_test = pickle.load(f)

# Define LSTM Model
model = Sequential([
    Embedding(input_dim=5000, output_dim=128, input_length=50),
    SpatialDropout1D(0.2),
    LSTM(100, dropout=0.2, recurrent_dropout=0.2),
    Dense(y_train.shape[1], activation='softmax')  # Softmax for multi-class classification
])

# Compile Model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train Model
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))

# Save Model
model.save("mood_lstm_model.h5")

print("Training complete. Model saved as mood_lstm_model.h5.")

