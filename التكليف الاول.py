import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

x = np.array([30, 5000, 4])

w = np.array([0.05, 0.002, 0.1]) 
b = -11.0  

z = np.dot(w, x) + b

y_hat = sigmoid(z)

decision = "Buy" if y_hat >= 0.5 else "Not Buy"

print("Homework Results:")
print("Customer:", x)
print("z value =", round(z, 4))
print("Probability =", round(y_hat, 4))
print("Decision =", decision)