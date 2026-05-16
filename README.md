```markdown
# Binary Number Neural Network
A neural network built from scratch using only NumPy that learns to predict numbers from their binary representation.

## Overview
This project implements a fully functional neural network without any ML libraries (TensorFlow, PyTorch, scikit-learn). Every component — forward propagation, backpropagation, gradient descent — is implemented manually.

## Architecture
```
Input (7 bits) → Hidden Layer (16 neurons, ReLU) → Output (1 neuron, linear)
```

## Components
| Function | Description |
|---|---|
| `transformer_bin(num)` | Converts integer to 7-bit binary list |
| `tranformer_int(binary)` | Converts binary list back to integer |
| `init_params(layer_sizes)` | Initializes weights and biases for each layer |
| `linear_layer(x, w, b)` | Computes z = Wx + b |
| `relu(x)` | ReLU activation function |
| `sigmoide(x)` | Sigmoid activation function |
| `forward_prop(x, params)` | Forward pass through all layers |
| `loss_mse(y_true, y_pred)` | MSE loss and its derivative |
| `back_prop(...)` | Backpropagation — computes gradients |
| `gradient_descent(lr, params, gradients)` | Updates weights using gradients |
| `train(...)` | Main training loop with early stopping |

## Training
- **Dataset**: integers 0–100, converted to 7-bit binary vectors
- **Split**: 80% train / 20% test (random, seeded for reproducibility)
- **Loss**: Mean Squared Error (MSE)
- **Optimizer**: Stochastic Gradient Descent (SGD)
- **Learning rate**: 0.0001

## Results
| Metric | Score |
|---|---|
| Exact accuracy | up to 100% |
| Within ±2 | up to 100% |

## Usage
```python
# train
params = init_params([7, 16, 1])
params = train(4000, x_train, y_train, params, lr=0.0001)

# predict
test = transformer_bin(42)
z, a = forward_prop(test, params)
print(round(z[-1][0]))  # → 42
```

## Key Concepts Implemented
- Linear layer (z = Wx + b)
- ReLU activation and its derivative
- Forward propagation
- Mean Squared Error loss
- Backpropagation with chain rule
- Stochastic Gradient Descent
- Early stopping with patience
```
