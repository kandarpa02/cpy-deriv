import numpy as np
from typing import Callable
from deriv.Array.reversed_mode_autodiff import _backward

def unbroadcast(grad, target_shape):
    """Reduces gradient to the original broadcasted shape."""
    while len(grad.shape) > len(target_shape):
        grad = grad.sum(axis=0)

    for i, dim in enumerate(target_shape):
        if dim == 1:
            grad = grad.sum(axis=i, keepdims=True)

    return grad

class array:
    def __init__(self, data, parents=(), need_grad=False):
        self.data = np.array(data) if not isinstance(data, np.ndarray) else data
        self.grad = np.zeros_like(self.data) if isinstance(self.data, list) else 0.0
        self._cached_topo = []
        self.shape = self.data.shape if isinstance(self.data, np.ndarray) else ()
        self.parents = parents
        def noop():
            pass
        self._back: Callable[[], None] = noop
        self.need_grad = need_grad
        self.grid_view:tuple = (5,6)
        self.is_scaler = True if self.data.shape == (1,1) else False

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]

    def backward(self):
        _backward(self)
    
    def __repr__(self):
        prefix = " " * len("array(")

        arr_str = np.array2string(
            self.data,
            precision=4,
            suppress_small=True,
            threshold=6,        
            edgeitems=3,       
            max_line_width=80, 
            separator=', ',     
            prefix=prefix     
        )

        extras = []
        if self._back.__name__ != "noop":
            extras.append(f"grad_fn=<{self._back.__name__}>")
        if self.need_grad:
            extras.append(f"need_grad={self.need_grad!r}")

        if extras:
            return f"array({arr_str}, " + ", ".join(extras) + ")"
        else:
            return f"array({arr_str})"


    def __add__(self, other):
        if isinstance(other, (int, float, list)):
            other = array(other)

        out = array(self.data + other.data, (self, other), need_grad=True)

        def add_back():
            if self.need_grad:
                self.grad += unbroadcast(out.grad, self.data.shape)

            if other.need_grad:
                other.grad += unbroadcast(out.grad, other.data.shape)

        out._back = add_back
        return out

    def __radd__(self, other):
        if isinstance(other, (int, float, list)):
            other = array(other)
        return other + self

    
    
    def __sub__(self, other):
        if isinstance(other, (int, float, list)):
            other = array(other)

        out = array(self.data - other.data, (self, other), need_grad=True)

        def sub_back():
            if self.need_grad:
                self.grad += unbroadcast(out.grad, self.data.shape)

            if other.need_grad:
                other.grad += unbroadcast(-out.grad, other.data.shape)

        out._back = sub_back
        return out
    
    def __rsub__(self, other):
        if isinstance(other, (int, float, list)):
            other = array(other)
        return other - self

    
    def __mul__(self, other):
        if isinstance(other, (int, float, list)):
            other = array(other)

        out = array(self.data * other.data, (self, other), need_grad=True)

        def mul_back():
            if self.need_grad:
                grad = other.data * out.grad
                self.grad += unbroadcast(grad, self.data.shape)

            if other.need_grad:
                grad = self.data * out.grad
                other.grad += unbroadcast(grad, other.data.shape)

        out._back = mul_back
        return out

    
    def __rmul__(self, other):
        if isinstance(other, (int, float, list)):
            other = array(other)
        return other * self


    def __truediv__(self, other):
        if isinstance(other, (int, float, list)):
            other = array(other)

        out = array(self.data / other.data, (self, other), need_grad=True)

        def div_back():
            if self.need_grad:
                grad_self = out.grad / other.data
                self.grad += unbroadcast(grad_self, self.data.shape)

            if other.need_grad:
                grad_other = -self.data * out.grad / (other.data ** 2)
                other.grad += unbroadcast(grad_other, other.data.shape)


        out._back = div_back
        return out
    

    def __rtruediv__(self, other):
        if isinstance(other, (int, float, list)):
            other = array(other)
        return other / self
    


    def __pow__(self, other):

        if isinstance(other, (int, float, list)):
            other = array(other)

        out = array(self.data ** other.data, (self, other), need_grad=True)

        def pow_back():
            if self.need_grad:
                grad_self = other.data * (self.data ** (other.data - 1)) * out.grad
                self.grad += unbroadcast(grad_self, self.data.shape)

            if other.need_grad:
                grad_other = out.data * np.log(self.data) * out.grad
                other.grad += unbroadcast(grad_other, other.data.shape)

        out._back = pow_back
        return out

    
    def __rpow__(self, other):
        if isinstance(other, (int, float, list)):
            other = array(other)
        return other ** self
        

    def __matmul__(self, other):
        if not isinstance(other, array):
            other = array(other)
        out = array(np.matmul(self.data, other.data), (self, other), need_grad=True)

        def matmul_back():
            if self.need_grad:
                self.grad += np.matmul(out.grad, np.swapaxes(other.data, -1, -2))
            if other.need_grad:
                other.grad += np.matmul(np.swapaxes(self.data, -1, -2), out.grad)

        out._back = matmul_back
        return out


    @property
    def T(self):
        out = array(self.data.T, (self,))  
        return out

    def sum(self, axis=None, keepdims=False):
        out = array(self.data.sum(axis=axis, keepdims=keepdims), (self,), need_grad=True)  

        def sumBackward():
            if self.need_grad:
                grad = out.grad
                if axis is not None and not keepdims:
                    shape = list(self.data.shape)
                    if isinstance(axis, int):
                        axis_ = [axis]
                    else:
                        axis_ = axis
                    for ax in axis_:
                        grad = np.expand_dims(grad, ax)
                self.grad += grad * np.ones_like(self.data)

        out._back = sumBackward
        return out

    
    def mean(self, axis=None):
        out = array(self.data.mean(axis=axis), (self,), need_grad=True)  

        def meanBackward():
            if self.need_grad:
                shape = self.data.shape
                if axis is None:
                    num_elements = np.prod(shape)
                    grad = np.ones_like(self.data) / num_elements
                    self.grad += out.grad * grad
                else:
                    num_elements = shape[axis]
                    grad = np.ones_like(self.data) / num_elements
                    self.grad += out.grad * grad 

        out._back = meanBackward
        return out


