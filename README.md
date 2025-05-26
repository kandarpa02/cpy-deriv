# deriv

deriv is a lightweight automatic differentiation library with a NumPy-like API and a minimal neural network framework. It is designed for clarity, extensibility, and low-level control over training loops and operations.

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
pip install git+https://github.com/kandarpa02/py-deriv.git
```

---
## Minimal Training Example

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

for epoch in range(20):
    pred = model(x)
    loss = ((pred - y) ** 2).sum()

    loss.backward()
    opt.step()
    opt.zero_grad()

    print(f"{loss.data:.6f}")

```
Now let us visualize the computation graph:

```python
loss.graph()
```
```bash
└── sum (0.1133774969574077)
    └── ** ([[0.1133775]])
        ├── - ([[0.33671575]])
        │   ├── + ([[1.33671575]])
        │   │   ├── @ ([[0.0494452]])
        │   │   │   ├──  ([[0.         0.14475168 0.1653283  0.28324537]])
        │   │   │   │   └── + ([[-0.0385508   0.14475168  0.1653283   0.28324537]])
        │   │   │   │       ├── @ ([[-0.0385508   0.14475168  0.1653283   0.28324537]])
        │   │   │   │       │   ├──  ([[1. 2.]])
        │   │   │   │       │   └──  ([[ 0.04319641  0.09602291  0.0445749   0.15318328]
 [-0.0408736   0.02436439  0.0603767   0.06503105]])
        │   │   │   │       └──  ([0. 0. 0. 0.])
        │   │   │   └──  ([[ 0.00022689]
 [ 0.092569  ]
 [-0.04119793]
 [ 0.15741855]])
        │   │   └──  ([1.30074148])
        │   └──  ([[1.]])
        └──  (2)
```

<details>
<summary>Sample Output</summary>

```bash
1.000934
0.960960
0.888020
0.790669
0.678023
0.558928
0.441305
0.331697
0.235007
0.154414
0.091441
0.046132
0.017320
0.002926
0.000281
0.006432
0.018406
0.033437
0.049123
0.063538
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
