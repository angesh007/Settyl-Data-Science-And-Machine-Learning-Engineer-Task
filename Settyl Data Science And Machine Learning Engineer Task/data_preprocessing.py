import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the dataset
with open('dataset.json', 'r') as file:
    data = json.load(file)

# Extract externalStatus and internalStatus
external_statuses = [entry['externalStatus'] for entry in data]
internal_statuses = [entry['internalStatus'] for entry in data]

# Encode internalStatus labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(internal_statuses)

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(external_statuses, encoded_labels, test_size=0.2, random_state=42)

# Inspect the processed data
print("X_train:", X_train)
print("y_train:", y_train)
print("X_val:", X_val)
print("y_val:", y_val)
