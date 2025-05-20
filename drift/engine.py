from drift.math.matrix_cpu import *
from typing import Callable
class tensor:
    def __init__(self, data, parents=(), need_grad=False):
        self.data = data
        self.grad = zeros_like_ct(self.data) if isinstance(self.data, list) else 0.0  # type: ignore
        self.shape = get_shape(self.data) if isinstance(self.data, list) else ()  # type: ignore
        self.parents = parents
        def noop():
            pass
        self._back: Callable[[], None] = noop
        self.need_grad = need_grad
        self.grid_view:tuple = (5,6)

    def __repr__(self):
        def format_element(e):
            if e == '...':
                return '...'
            e_float = float(e)
            rounded = round(e_float, 4)
            if rounded.is_integer():
                return f"{int(rounded)}."
            else:
                return "{:.2f}".format(rounded).rstrip('0').rstrip('.') if '.' in "{:.2f}".format(rounded) else "{:.2f}".format(rounded)

        def format_data(data, depth=0):
            max_rows, max_cols = self.grid_view
            if isinstance(data, list):
                if all(isinstance(row, list) for row in data):
                    num_rows = len(data)
                    if depth == 0 and num_rows > max_rows:
                        half = max_rows // 2
                        head = data[:half]
                        tail = data[-half:]
                        truncated = head + [['...']] + tail
                    else:
                        truncated = data

                    formatted_rows = []
                    for row in truncated:
                        if row == ['...']:
                            formatted_rows.append(['...'])
                            continue
                        num_cols = len(row)
                        if num_cols > max_cols:
                            half_col = max_cols // 2
                            head_col = row[:half_col]
                            tail_col = row[-half_col:]
                            row_truncated = head_col + ['...'] + tail_col
                        else:
                            row_truncated = row
                        formatted_row = [format_element(e) for e in row_truncated]
                        formatted_rows.append(formatted_row)

                    num_cols = max(len(row) for row in formatted_rows) if formatted_rows else 0
                    col_widths = [0] * num_cols
                    for row in formatted_rows:
                        for j in range(len(row)):
                            col_widths[j] = max(col_widths[j], len(row[j]))

                    padded_rows = []
                    for row in formatted_rows:
                        padded = []
                        for j in range(len(row)):
                            elem = row[j]
                            if elem == '...':
                                padded.append(elem.ljust(col_widths[j]))
                            else:
                                padded.append(elem.rjust(col_widths[j]))
                        padded_row = "[{}]".format(", ".join(padded))
                        padded_rows.append(padded_row)
                    return "[{}]".format(",\n        ".join(padded_rows))
                else:
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
                    padded = [e.rjust(max_width) if e != '...' else e.ljust(max_width) for e in elements]
                    return "[{}]".format(", ".join(padded))
            else:
                return format_element(data)

        data_str = format_data(self.data)
        _grad_fn = (
            f", grad_fn=<{self._back.__name__}>" 
            if self._back.__name__ != "noop" 
            else ""
        )
        _need_grad = f", need_grad={self.need_grad}" if self.need_grad else ""
        if _need_grad != "" and _grad_fn == "":
            return f"tensor({data_str}{_need_grad})"
        return f"tensor({data_str}{_grad_fn})"
    

    def backward(self, grad=None):
        if grad is None:
            if isinstance(self.data, list):
                if not (len(self.data) == 1 and len(self.data[0]) == 1):
                    raise RuntimeError("grad can be implicitly created only for scalar outputs. "
                                    "Use `backward(grad)` to supply the gradient manually.")
                grad = [[1.0]]
            else:
                grad = 1.0

        # If grad is a tensor object, extract the raw data
        self.grad = grad.data if isinstance(grad, tensor) else grad

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
            other = tensor(other)

        added = addition(self.data, other.data) # type: ignore

        out = tensor(added, (self, other))

        def addBackward():
            self.grad = addition(self.grad, reduce_grad(out.grad, self.shape)) # type: ignore
            other.grad = addition(other.grad, reduce_grad(out.grad, other.shape)) # type: ignore

        out._back = addBackward
        return out

    def __radd__(self, other): 
        return self.__add__(other)
    
    
    def __mul__(self, other):
        if isinstance(other, (int, float, list)):
            other = tensor(other)

        multiplied = multiplication(self.data, other.data) # type: ignore

        out = tensor(multiplied, (self, other))

        def mulBackward():
                
            grad_out = out.grad

            grad_self_matrix = multiplication(other.data, grad_out) # type: ignore

            if self.shape == ():
                grad_self = array_sum(grad_self_matrix, axis=None) # type: ignore
            else:
                grad_self = reduce_grad(grad_self_matrix, self.shape) # type: ignore

            self.grad = addition(self.grad, grad_self) # type: ignore

            grad_other_matrix = multiplication(self.data, grad_out) # type: ignore

            if other.shape == ():  # scalar case
                grad_other = array_sum(grad_other_matrix, axis=None) # type: ignore
            else:
                grad_other = reduce_grad(grad_other_matrix, other.shape) # type: ignore

            other.grad = addition(other.grad, grad_other) # type: ignore


        out._back = mulBackward
        return out
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    
    def __matmul__(self, other):
        out = tensor(matmul(self.data, other.data), (self, other)) # type: ignore

        def matmulBackward():
            self.grad = addition(self.grad, matmul(out.grad, transpose(other.data)))  # type: ignore
            other.grad = addition(other.grad, matmul(transpose(self.data), out.grad)) # type: ignore

        out._back = matmulBackward
        return out

    @property
    def T(self):
        out = tensor(transpose(self.data), (self,)) # type: ignore
        return out

    def sum(self, axis=None):
        out = tensor(array_sum(self.data, axis), (self,)) # type: ignore

        def sumBackward():
            self.grad = multiplication(out.grad, ones_like_ct(self.data)) # type: ignore
            
        out._back = sumBackward
        return out


def ones_like(A):
    if isinstance(A, tensor):
        A = A.data
    return tensor(ones_like_ct(A)) # type: ignore

def zeros_like(A):
    if isinstance(A, tensor):
        A = A.data
    return tensor(zeros_like_ct(A)) # type: ignore

