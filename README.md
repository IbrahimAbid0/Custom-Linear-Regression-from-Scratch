# 📈 Custom Linear Regression from Scratch

This project implements a **Linear Regression** model in Python **from scratch**, using `numpy` only . The implementation is encapsulated in a class called `LinearRegression` with functions for training, prediction, and accessing model weights.

---

## 📌 Features

- Implements **Gradient Descent** for optimization
- Supports **custom learning rate** and **early stopping**
- Prints loss during training at configurable intervals
- Designed for **educational purposes** to demonstrate how linear regression works under the hood

---

## 🧠 Class Overview

### `LinearRegression`

| Function         | Description |
|------------------|-------------|
| `train(X, Y, alpha=0.01, max_iters=None, print_loss_every=100)` | Trains the model using gradient descent. If `max_iters` is not provided, an internal stopping criterion is used. |
| `predict(X)`     | Returns predicted outputs for the input `X`. |
| `get_weights()`  | Returns the weights (`w`) and bias (`b`) of the trained model. |

---

## 🧪 Example Usage

```python
from linear_regression import LinearRegression

# Sample training data
X = [[1], [2], [3], [4]]
Y = [2, 4, 6, 8]

# Initialize and train
model = LinearRegression()
model.train(X, Y, alpha=0.01, max_iters=1000, print_loss_every=100)

# Make predictions
predictions = model.predict([[5], [6]])
print("Predictions:", predictions)

# Get learned weights
w, b = model.get_weights()
print("Weights:", w, "Bias:", b)
