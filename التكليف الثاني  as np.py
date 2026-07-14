import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

# For reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# 1. Load the Dataset
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# 2. Explore the Data
print("Number of samples:", df.shape[0])
print("Number of features:", df.shape[1] - 1)
print("Target classes:", data.target_names)
print("\nFirst five rows:")
print(df.head())

# 3. Prepare the Data
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# 4. Build an MLP Model
model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(16, activation='relu'),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.summary()

# 5. Compile the Model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 6. Train the Model
history = model.fit(
    X_train, 
    y_train, 
    epochs=50, 
    batch_size=16, 
    validation_split=0.2, 
    verbose=1
)

# 7. Evaluate the Model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Loss: {round(loss, 4)}")
print(f"Test Accuracy: {round(accuracy, 4)}")

# Plot training loss
plt.figure(figsize=(7, 5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("MLP Model Loss Progression")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()