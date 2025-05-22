from typing import Callable
import numpy as np

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
        self.shape = self.data.shape if isinstance(self.data, np.ndarray) else ()
        self.parents = parents
        def noop():
            pass
        self._back: Callable[[], None] = noop
        self.need_grad = need_grad
        self.grid_view:tuple = (5,6)
        self.is_scaler = True if self.data.shape == (1,1) else False
    
    def __repr__(self):
        def format_element(e):
            # Handle array-like elements (e.g., numpy scalars)
            if hasattr(e, 'item'):  # Convert numpy scalars to Python floats
                e = e.item()
            if e == '...':
                return '...'
            e_float = float(e)
            rounded = round(e_float, 4)
            if rounded.is_integer():
                return f"{int(rounded)}."
            else:
                formatted = "{:.4f}".format(rounded)
                stripped = formatted.rstrip('0').rstrip('.')
                return stripped if stripped else formatted

        def format_data(data, depth=0):
            max_rows, max_cols = (4, 6)  # Truncation settings

            # Convert array-like data to lists
            if hasattr(data, 'tolist'):
                data = data.tolist()

            if isinstance(data, list):
                if all(isinstance(elem, (list, np.ndarray)) for elem in data):
                    # Handle nested lists (2D+)
                    num_elements = len(data)
                    if num_elements > max_rows:
                        half = max_rows // 2
                        head = data[:half]
                        tail = data[-half:]
                        truncated = head + [['...']] + tail
                    else:
                        truncated = data

                    formatted = []
                    for elem in truncated:
                        if elem == ['...']:
                            formatted.append('...')
                        else:
                            formatted_elem = format_data(elem, depth + 1)
                            formatted.append(formatted_elem)

                    indent = ' ' * 4 * (depth + 1)
                    separator = ",\n" + indent
                    return f"[{separator.join(formatted)}]"
                else:
                    # Handle 1D list
                    num_elements = len(data)
                    if num_elements > max_cols:
                        half = max_cols // 2
                        head = data[:half]
                        tail = data[-half:]
                        truncated = head + ['...'] + tail
                    else:
                        truncated = data

                    elements = [format_element(e) for e in truncated]
                    max_width = max(len(e) for e in elements) if elements else 0
                    padded = [
                        e.rjust(max_width) if e != '...' else e.ljust(max_width)
                        for e in elements
                    ]
                    return "[{}]".format(", ".join(padded))
            else:
                return format_element(data)

        data_str = format_data(self.data)
        grad_fn = f", grad_fn=<{self._back.__name__}>" if self._back.__name__ != "noop" else ""
        need_grad = f", need_grad={self.need_grad}" if self.need_grad else ""
        
        parts = [f"array({data_str}"]
        if grad_fn:
            parts.append(grad_fn)
        if need_grad:
            parts.append(need_grad)
        parts.append(")")
        
        return "".join(parts)

    def backward(self):
        self.grad = np.ones_like(self.data) if self.data.shape != (1,) else 1.0

        visited = set()
        topo = []

        def build_topo(node):
            if node not in visited:
                visited.add(node)
                for parent in node.parents:
                    build_topo(parent)
                topo.append(node)

        build_topo(self)

        for node in reversed(topo):
            node._back()


    def __add__(self, other):
        if isinstance(other, (int, float, list)):
            other = array(other)

        out = array(self.data + other.data, (self, other), need_grad=True)

        def addBackward():
            if self.need_grad:
                self.grad += unbroadcast(out.grad, self.data.shape)

            if other.need_grad:
                other.grad += unbroadcast(out.grad, other.data.shape)

        out._back = addBackward
        return out

    def __radd__(self, other):
        if isinstance(other, (int, float, list)):
            other = array(other)
        return other + self

    
    
    def __sub__(self, other):
        if isinstance(other, (int, float, list)):
            other = array(other)

        out = array(self.data - other.data, (self, other), need_grad=True)

        def subBackward():
            if self.need_grad:
                self.grad += unbroadcast(out.grad, self.data.shape)

            if other.need_grad:
                other.grad += unbroadcast(-out.grad, other.data.shape)

        out._back = subBackward
        return out
    
    def __rsub__(self, other):
        if isinstance(other, (int, float, list)):
            other = array(other)
        return other - self

    
    def __mul__(self, other):
        if isinstance(other, (int, float, list)):
            other = array(other)

        out = array(self.data * other.data, (self, other), need_grad=True)

        def mulBackward():
            if self.need_grad:
                grad = other.data * out.grad
                self.grad += unbroadcast(grad, self.data.shape)

            if other.need_grad:
                grad = self.data * out.grad
                other.grad += unbroadcast(grad, other.data.shape)

        out._back = mulBackward
        return out

    
    def __rmul__(self, other):
        if isinstance(other, (int, float, list)):
            other = array(other)
        return other * self


    def __truediv__(self, other):
        if isinstance(other, (int, float, list)):
            other = array(other)

        out = array(self.data / other.data, (self, other), need_grad=True)

        def divBackward():
            if self.need_grad:
                grad_self = out.grad / other.data
                self.grad += unbroadcast(grad_self, self.data.shape)

            if other.need_grad:
                grad_other = -self.data * out.grad / (other.data ** 2)
                other.grad += unbroadcast(grad_other, other.data.shape)


        out._back = divBackward
        return out
    
    def __rtruediv__(self, other):
        if isinstance(other, (int, float, list)):
            other = array(other)
        return other / self
    


    def __pow__(self, other):

        if isinstance(other, (int, float, list)):
            other = array(other)

        out = array(self.data ** other.data, (self, other), need_grad=True)

        def powBackward():
            if self.need_grad:
                grad_self = other.data * (self.data ** (other.data - 1)) * out.grad
                self.grad += unbroadcast(grad_self, self.data.shape)

            if other.need_grad:
                grad_other = out.data * np.log(self.data) * out.grad
                other.grad += unbroadcast(grad_other, other.data.shape)

        out._back = powBackward
        return out

    
    def __rpow__(self, other):
        if isinstance(other, (int, float, list)):
            other = array(other)
        return other ** self



    def __matmul__(self, other):
        out = array(np.matmul(self.data, other.data), (self, other), need_grad=True)

        def matmulBackward():
            if self.need_grad:
                self.grad += np.matmul(out.grad, np.swapaxes(other.data, -1, -2))
            if other.need_grad:
                other.grad += np.matmul(np.swapaxes(self.data, -1, -2), out.grad)

        out._back = matmulBackward
        return out



    @property
    def T(self):
        out = array(self.data.T, (self,))  
        return out

    def sum(self, axis=None):
        out = array(self.data.sum(), (self,))  

        def sumBackward():
            if self.need_grad:
                self.grad += out.grad * np.ones_like(self.data) 
            
        out._back = sumBackward
        return out

