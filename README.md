# PyDeriv

PyDeriv is a lightweight automatic differentiation library with a NumPy-like API and a minimal neural network framework. It is designed for clarity, extensibility, and low-level control over training loops and operations.

> This is an early alpha release. APIs may change.

---
## Features
- Reverse-mode autodiff from scratch

- NumPy-like API and syntax

- Basic neural layers: Dense, ReLU, Tanh

- Custom optimizer support

- Pure Python with minimal dependencies

---
## Installation

```bash
pip install git+https://github.com/kandarpa02/PyDeriv.git
```

---
## 🚀 Minimal Training Example

```python
import deriv
from deriv.nn import dense, ReLU
from deriv.optim import SGD
from deriv import array

# Define a simple model
class MLP:
    def __init__(self):
        self.fc1 = dense(2, 4)
        self.act = ReLU()
        self.fc2 = dense(4, 1)

    def __call__(self, x):
        x = self.fc2(self.act(self.fc1(x)))
        return x

# Training loop
model = MLP()
opt = SGD(parameters=model.fc1.parameters() | model.fc2.parameters(), lr=0.1)

x = array([[1.0, 2.0]])
y = array([[1.0]])

for epoch in range(10):
    pred = model(x)
    loss = ((pred - y) ** 2).sum()

    loss.backward()
    opt.step()
    opt.zero_grad()

    print(f"{loss.data:.6f}")

```
<details>
<summary>📉 Sample Output</summary>
```
1.000834
0.959335
0.883733
0.783097
0.667094
0.545077
0.425373
0.314792
0.218350
0.139192
```
</details>

## Autodiff in Action

Here's a sample forward and backward pass using Deriv:

![Deriv autodiff demo](assets/deriv_matmul.png)

The above code computes a composite function:

```
z = xy + x + 2 
h = ReLU(z) + x^2 
q = tanh(h) + 3y
f = q^2 + (z - h) + (A @ B)
```

Where:
- x = -3
- y = 1
- A = [2, -1]
- B = [[3], [1]]


This produces:
```
f = 8.0
df/dx = 42.0000
df/dy = -3.0000
df/dA = [3.0, 1.0]
df/dB = [[2.0], [-1.0]]
```

> Deriv handles broadcasting, ReLU/Tanh activation, matrix multiplication, and complex chaining — all from scratch.

---

## Disclaimer

Still under heavy development. Expect breaking changes. For now it only works on CPU.
