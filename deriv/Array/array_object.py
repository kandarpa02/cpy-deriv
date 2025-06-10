from typing import Callable
from deriv.Array.reversed_mode_autodiff import _backward
from deriv.Array.backend import get_backend


def unbroadcast(grad, target_shape):
    """Reduces gradient to the original broadcasted shape."""
    while len(grad.shape) > len(target_shape):
        grad = grad.sum(axis=0)

    for i, dim in enumerate(target_shape):
        if dim == 1:
            grad = grad.sum(axis=i, keepdims=True)

    return grad

class array:
    """
    deriv.array(data, parents=(), need_grad=False)

    A NumPy-compatible array class with reverse-mode autodiff support.

    Parameters
    ----------
    data : array_like
        Input data to initialize the array.
    parents : tuple, optional
        The parent nodes in the computation graph, used for backpropagation.
    need_grad : bool, optional
        Whether to track gradients for this array.
    """
    
    def __init__(self, data, parents=(), op='', need_grad=False, var_name=''):
        self.xp = get_backend()
        self.data = self.xp.array(data) if not isinstance(data, self.xp.ndarray) else data
        self.grad = self.xp.zeros_like(self.data) if need_grad else None
        self._cached_topo = []
        self.shape = self.data.shape if isinstance(self.data, self.xp.ndarray) else ()
        self.parents = parents
        self.op = op
        def noop():
            pass
        self._back: Callable[[], None] = noop
        self.need_grad = need_grad
        self.var_name = var_name
        self.is_scaler = True if self.data.shape == (1,1) else False

    @property
    def dtype(self):
        return type(self)

    def backward(self):
        """
        deriv.backward()

        Computes the gradient of the array with respect to all `need_grad=True` inputs.
        """
        _backward(self)

    def topo(self):
        if not self._cached_topo:  
            visited = set()
            def build_topo(node):
                if node not in visited:
                    visited.add(node)
                    for parent in node.parents:
                        build_topo(parent)
                    self._cached_topo.append(node)
            build_topo(self)
        return self._cached_topo

    def graph(self, data=False):
        def print_graph(node, indent="", last=True, visited=None):
            if visited is None:
                visited = set()
                
            if node in visited:
                print(indent + ("└── " if last else "├── ") + f"[...]{get_node_label(node)}")
                return
            visited.add(node)

            branch = "└── " if last else "├── "
            print(indent + branch + get_node_label(node))

            indent += "    " if last else "│   "
            for i, parent in enumerate(node.parents):
                is_last = (i == len(node.parents) - 1)
                print_graph(parent, indent, is_last, visited)

        def get_node_label(node):
            def get_last_node(self): 
                topo = []
                visited = set()
                def build_topo(node):
                    if node not in visited:
                        visited.add(node)
                        for parent in node.parents:
                            build_topo(parent)
                        topo.append(node)
                build_topo(self)
                topo.reverse()
                return topo
            
            topo = get_last_node(self)

            if node == topo[0] and hasattr(node, 'op'):
                return f"{node.op} ({node.data})"
            if node.var_name:
                if data == False:
                    return f"{node.var_name}"
                return f"{node.var_name} ({node.data})"
            elif hasattr(node, 'op'):
                if data == False:
                    return f"{node.op}"
                return f"{node.op} ({node.data})"
            elif isinstance(node, (float, int)):
                return f"(node)"
            else:
                return f"val ({node.data})"
            
        return print_graph(self)

    def __repr__(self):
        """
        String representation of the deriv.array object.
        """
        prefix = " " * len("array(")
        arr_str = self.xp.array2string(
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
        if self.var_name != '':
            extras.append(f"variable={self.var_name}")
        if extras:
            return f"array({arr_str}, " + ", ".join(extras) + ")"
        else:
            return f"array({arr_str})"

    def __add__(self, other):
        """
        deriv.add(self, other)

        Element-wise addition.
        """
        if isinstance(other, (int, float, list)):
            other = array(other)
        out = array(self.data + other.data, (self, other), '+', need_grad=True)
        def add_back():
            if self.need_grad:
                self.grad += unbroadcast(out.grad, self.data.shape)
            if other.need_grad:
                other.grad += unbroadcast(out.grad, other.data.shape)
        out._back = add_back
        return out

    def __radd__(self, other):
        """Reflected addition."""
        if isinstance(other, (int, float, list)):
            other = array(other)
        return other + self

    def __sub__(self, other):
        """
        deriv.sub(self, other)

        Element-wise subtraction.
        """
        if isinstance(other, (int, float, list)):
            other = array(other)
        out = array(self.data - other.data, (self, other), '-', need_grad=True)
        def sub_back():
            if self.need_grad:
                self.grad += unbroadcast(out.grad, self.data.shape)
            if other.need_grad:
                other.grad += unbroadcast(-out.grad, other.data.shape)
        out._back = sub_back
        return out

    def __rsub__(self, other):
        """Reflected subtraction."""
        if isinstance(other, (int, float, list)):
            other = array(other)
        return other - self

    def __mul__(self, other):
        """
        deriv.mul(self, other)

        Element-wise multiplication.
        """
        if isinstance(other, (int, float, list)):
            other = array(other)
        out = array(self.data * other.data, (self, other), '*', need_grad=True)
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
        """Reflected multiplication."""
        if isinstance(other, (int, float, list)):
            other = array(other)
        return other * self

    def __truediv__(self, other):
        """
        deriv.truediv(self, other)

        Element-wise division.
        """
        if isinstance(other, (int, float, list)):
            other = array(other)
        out = array(self.data / other.data, (self, other), '/', need_grad=True)
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
        """Reflected division."""
        if isinstance(other, (int, float, list)):
            other = array(other)
        return other / self

    def __pow__(self, other):
        """
        deriv.pow(self, other)

        Element-wise exponentiation.
        """
        if isinstance(other, (int, float, list)):
            other = array(other)
        out = array(self.data ** other.data, (self, other), '**', need_grad=True)
        def pow_back():
            if self.need_grad:
                grad_self = other.data * (self.data ** (other.data - 1)) * out.grad
                self.grad += unbroadcast(grad_self, self.data.shape)
            if other.need_grad:
                grad_other = out.data * self.xp.log(self.data) * out.grad
                other.grad += unbroadcast(grad_other, other.data.shape)
        out._back = pow_back
        return out

    def __rpow__(self, other):
        """Reflected exponentiation."""
        if isinstance(other, (int, float, list)):
            other = array(other)
        return other ** self

    def __matmul__(self, other):
        """
        deriv.matmul(self, other)

        Matrix multiplication.
        """
        if not isinstance(other, array):
            other = array(other)
        out = array(self.xp.matmul(self.data, other.data), (self, other), '@', need_grad=True)
        def matmul_back():
            if self.need_grad:
                self.grad += self.xp.matmul(out.grad, self.xp.swapaxes(other.data, -1, -2))
            if other.need_grad:
                other.grad += self.xp.matmul(self.xp.swapaxes(self.data, -1, -2), out.grad)
        out._back = matmul_back
        return out

    @property
    def T(self):
        """
        deriv.T

        Returns the transpose of the array.
        """
        out = array(self.data.T, (self,), op='T')  
        return out

    def sum(self, axis=None, keepdims=False):
        """
        deriv.sum(self, axis=None, keepdims=False)

        Sum of array elements over a given axis.

        Parameters
        ----------
        axis : None or int or tuple of ints, optional
            Axis or axes along which a sum is performed.
        keepdims : bool, optional
            If True, retains reduced dimensions with size one.
        """
        out = array(self.data.sum(axis=axis, keepdims=keepdims), (self,), 'sum', need_grad=True)  
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
                        grad = self.xp.expand_dims(grad, ax)
                self.grad += grad * self.xp.ones_like(self.data)
        out._back = sumBackward
        return out

    def mean(self, axis=None):
        """
        deriv.mean(self, axis=None)

        Mean of array elements over a given axis.

        Parameters
        ----------
        axis : None or int, optional
            Axis or axes along which the mean is computed.
        """
        out = array(self.data.mean(axis=axis), (self,), 'mean', need_grad=True)  
        def meanBackward():
            if self.need_grad:
                shape = self.data.shape
                if axis is None:
                    num_elements = self.xp.prod(shape)
                    grad = self.xp.ones_like(self.data) / num_elements
                    self.grad += out.grad * grad
                else:
                    num_elements = shape[axis]
                    grad = self.xp.ones_like(self.data) / num_elements
                    self.grad += out.grad * grad 
        out._back = meanBackward
        return out
    
    def max(self, axis=None, keepdims=False):
        out = array(self.data.max(axis=axis, keepdims=keepdims), (self,), 'max', need_grad=True)
        return out
        

    def __len__(self):
        """Returns the number of elements along the first axis."""
        return len(self.data)

    def __getitem__(self, index):
        """Indexing access to data."""
        return self.data[index]

    def __eq__(self, other):
        """Equality check (reference based)."""
        return self is other

    def __ne__(self, other):
        """Non-equality check."""
        if isinstance(other, self.__class__):
            return self.data != other.data
        return NotImplemented

    def __lt__(self, other):
        """Less-than comparison."""
        if isinstance(other, self.__class__):
            return self.data < other.data
        return NotImplemented

    def __le__(self, other):
        """Less-than or equal comparison."""
        if isinstance(other, self.__class__):
            return self.data <= other.data
        return NotImplemented

    def __gt__(self, other):
        """Greater-than comparison."""
        if isinstance(other, self.__class__):
            return self.data > other.data
        return NotImplemented

    def __ge__(self, other):
        """Greater-than or equal comparison."""
        if isinstance(other, self.__class__):
            return self.data >= other.data
        return NotImplemented

    def __hash__(self):
        """Object hash based on id."""
        return id(self)
    
    def __neg__(self):
        return array(-self.data)

    def to(self, device: str):
        if device == 'cpu':
            import numpy as np
            if self.data.__class__.__module__.startswith('cupy'):
                import cupy as cp
                self.data =  cp.asnumpy(self.data)
            self.device = 'cpu'
        elif device == 'cuda':
            import cupy as cp
            if self.data.__class__.__module__.startswith('numpy'):
                self.data = cp.asarray(self.data)
            self.device = 'cuda'
        else:
            raise ValueError("Device must be 'cpu' or 'cuda'")
        return self  # enable chaining