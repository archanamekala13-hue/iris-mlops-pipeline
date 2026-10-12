
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Load the dataset
data = pd.read_csv("iris.csv")

print("Total dataset rows:", len(data))

# 2. Separate inputs and outputs
X = data.drop("species", axis=1)
y = LabelEncoder().fit_transform(data["species"])

# 3. Split data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 4. Scale the measurements
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. Create the neural network
model = MLPClassifier(
    hidden_layer_sizes=(10, 10),
    activation="relu",
    solver="lbfgs",
    max_iter=5000,
    random_state=42
)

# 6. Train the model
model.fit(X_train, y_train)

# 7. Make predictions on unseen test data
y_pred = model.predict(X_test)

# 8. Display results
accuracy = accuracy_score(y_test, y_pred)

print("\nNeural Network Results")
print("----------------------")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print(f"Test Accuracy: {accuracy * 100:.2f}%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=LabelEncoder().fit(data["species"]).classes_
))
