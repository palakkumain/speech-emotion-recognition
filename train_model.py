# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Step 1: Load the extracted features
print(" Loading extracted features...")
data = pd.read_csv("features.csv")

# Step 2: Separate features and labels
X = data.drop(columns=['emotion'])
y = data['emotion']

# Step 3: Encode emotion labels (text → numeric)
le = LabelEncoder()
y = le.fit_transform(y)

# Step 4: Split dataset into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Feature scaling (standardize input features)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 6: Train a Random Forest Classifier
print("Training Random Forest model...")
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Step 7: Evaluate the model
y_pred = model.predict(X_test)

print("\nModel Training Completed!")
print(f" Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\n Classification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))
print("\n Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Step 8: Save the model and label encoder for future predictions
joblib.dump(model, "emotion_model.pkl")
joblib.dump(le, "label_encoder.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\n Model, Label Encoder, and Scaler saved successfully!")
