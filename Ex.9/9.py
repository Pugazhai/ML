import numpy as np
import matplotlib.pyplot as plt

def lwr(query_point, X, y, tau):
    """
    Locally Weighted Regression
    Args:
    - query_point: point at which prediction is to be made
    - X: input features
    - y: target values
    - tau: bandwidth parameter
    Returns:
    - prediction at query_point
    """
    m = X.shape[0]
    X = np.column_stack((np.ones(m), X))  # Add bias term
    query_point = np.array([1, query_point])  # Add bias term to query point
    weights = np.exp(-((X[:, 1] - query_point[1]) ** 2) / (2 * tau * tau))
    W = np.diag(weights)
    theta = np.linalg.inv(X.T @ W @ X) @ (X.T @ (W @ y))
    prediction = query_point @ theta
    return prediction

np.random.seed(0)
X = np.linspace(0, 10, 100)
y = np.sin(X) + np.random.normal(0, 0.1, 100)

tau = 1.0

predictions = [lwr(x, X, y, tau) for x in X]

plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Original Data')
plt.plot(X, predictions, color='red', label='Fitted Curve')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Locally Weighted Regression')
plt.legend()
plt.show()
