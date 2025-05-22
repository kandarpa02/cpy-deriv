# Deriv 

Deriv is a highly efficient autodiff library with a NumPy-like API.

This is an early alpha release. APIs will change.

## Features

- Automatic gradient engine
- Efficient matrix ops
- NumPy-style syntax

---

## 🧪 Autodiff in Action

Here's a sample forward and backward pass using Deriv:

![Deriv autodiff demo](assets/deriv_matmul.png)

The above code computes a composite function:

\[
\begin{align*}
z &= xy + x + 2 \\
h &= \text{ReLU}(z) + x^2 \\
q &= \tanh(h) + 3y \\
f &= q^2 + (z - h) + (A \cdot B)
\end{align*}
\]

Where:
- \( x = -3 \), \( y = 1 \)
- \( A = [2, -1] \), \( B = \begin{bmatrix} 3 \\ 1 \end{bmatrix} \)

This produces:
- \( f = 8.0 \)
- \( \frac{\partial f}{\partial x} = 42.0000 \)
- \( \frac{\partial f}{\partial y} = -3.0000 \)
- \( \frac{\partial f}{\partial A} = [3.0, 1.0] \)
- \( \frac{\partial f}{\partial B} = \begin{bmatrix} 2.0 \\ -1.0 \end{bmatrix} \)

> 🔥 Deriv handles broadcasting, ReLU/Tanh activation, matrix multiplication, and complex chaining — all from scratch.

---

## 🚧 Disclaimer

Still under heavy development. Expect breaking changes. For now it only works on CPU.
