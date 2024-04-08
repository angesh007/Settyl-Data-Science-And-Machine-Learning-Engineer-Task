import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

# Load the dataset
with open('dataset.json', 'r') as f:
    dataset = json.load(f)

# Extract external and internal statuses
external_statuses = [data['externalStatus'] for data in dataset]
internal_statuses = [data['internalStatus'] for data in dataset]

# Encode internal statuses
label_encoder = LabelEncoder()
encoded_internal_statuses = label_encoder.fit_transform(internal_statuses)

# Split the dataset into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(external_statuses, encoded_internal_statuses, test_size=0.2, random_state=42)

# Tokenize the external statuses
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
vocab_size = len(tokenizer.word_index) + 1
max_length = max([len(status.split()) for status in X_train])

# Convert text data to sequences
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_val_seq = tokenizer.texts_to_sequences(X_val)

# Pad sequences for equal length
X_train_padded = pad_sequences(X_train_seq, maxlen=max_length, padding='post')
X_val_padded = pad_sequences(X_val_seq, maxlen=max_length, padding='post')

# Define the LSTM model
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=100, input_length=max_length))
model.add(LSTM(units=64, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(units=np.max(encoded_internal_statuses) + 1, activation='softmax'))

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Print model summary
print(model.summary())

# Train the model
history = model.fit(X_train_padded, y_train, epochs=10, batch_size=64, validation_data=(X_val_padded, y_val))

# Evaluate the model
loss, accuracy = model.evaluate(X_val_padded, y_val)
print("Validation Accuracy:", accuracy)

# Predictions on validation set
predictions = model.predict(X_val_padded)

# Convert predictions to class labels
y_pred = np.argmax(predictions, axis=1)

# Decode internal status labels
decoded_internal_statuses = label_encoder.inverse_transform(y_val)
decoded_predictions = label_encoder.inverse_transform(y_pred)

# Print classification report
print(classification_report(decoded_internal_statuses, decoded_predictions))
