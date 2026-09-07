import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, SimpleRNN, Dense

df = pd.read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv')

print(df.head())
print(df.info())
print(df.isnull().sum())
print("Total records:", len(df))

df['Date'] = pd.to_datetime(df['Date'])
df['Temp'] = pd.to_numeric(df['Temp'], errors='coerce')
df = df.dropna()

plt.figure(figsize=(12, 5))
plt.plot(df['Date'], df['Temp'])
plt.title('Daily Minimum Temperatures Over Time')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.show()

data = df['Temp'].values.reshape(-1, 1)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

def create_sequences(dataset, time_steps=7):
    X, y = [], []
    for i in range(len(dataset) - time_steps):
        X.append(dataset[i:(i + time_steps), 0])
        y.append(dataset[i + time_steps, 0])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data, time_steps=7)

train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

model = Sequential([
    Input(shape=(7, 1)),
    SimpleRNN(32, activation='tanh'),
    Dense(1)
])

model.summary()

model.compile(optimizer='adam', loss='mean_squared_error')
history = model.fit(X_train, y_train, epochs=25, batch_size=32, validation_split=0.1)

plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training Loss vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.show()

predictions = model.predict(X_test)
predictions_rescaled = scaler.inverse_transform(predictions)
y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

mae = mean_absolute_error(y_test_rescaled, predictions_rescaled)
mse = mean_squared_error(y_test_rescaled, predictions_rescaled)
print("MAE:", mae)
print("MSE:", mse)

plt.figure(figsize=(12, 5))
plt.plot(y_test_rescaled, label='Actual Temperature')
plt.plot(predictions_rescaled, label='Predicted Temperature')
plt.title('Actual Temperature vs Predicted Temperature')
plt.xlabel('Time Steps')
plt.ylabel('Temperature')
plt.legend()
plt.show()

model_A = Sequential([
    Input(shape=(7, 1)),
    SimpleRNN(16, activation='tanh'),
    Dense(1)
])
model_A.compile(optimizer='adam', loss='mean_squared_error')
history_A = model_A.fit(X_train, y_train, epochs=25, batch_size=32, verbose=0)

model_B = Sequential([
    Input(shape=(7, 1)),
    SimpleRNN(32, activation='tanh'),
    Dense(1)
])
model_B.compile(optimizer='adam', loss='mean_squared_error')
history_B = model_B.fit(X_train, y_train, epochs=25, batch_size=32, verbose=0)

pred_A = scaler.inverse_transform(model_A.predict(X_test))
pred_B = scaler.inverse_transform(model_B.predict(X_test))

mae_A = mean_absolute_error(y_test_rescaled, pred_A)
mse_A = mean_squared_error(y_test_rescaled, pred_A)

mae_B = mean_absolute_error(y_test_rescaled, pred_B)
mse_B = mean_squared_error(y_test_rescaled, pred_B)

print(f"Model A (16 units) -> Loss: {history_A.history['loss'][-1]:.5f}, MAE: {mae_A:.3f}, MSE: {mse_A:.3f}")
print(f"Model B (32 units) -> Loss: {history_B.history['loss'][-1]:.5f}, MAE: {mae_B:.3f}, MSE: {mse_B:.3f}")

plt.figure(figsize=(12, 5))
plt.plot(y_test_rescaled[:100], label='Actual')
plt.plot(pred_A[:100], label='Model A (16 units)')
plt.plot(pred_B[:100], label='Model B (32 units)')
plt.title('Comparison: Model A vs Model B (First 100 Test Steps)')
plt.xlabel('Time Steps')
plt.ylabel('Temperature')
plt.legend()
plt.show()