import numpy as np
from deriv.Array.array_object import array
from deriv.nn.module import Parameter, Module

class dense(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        w = array(np.random.randn(in_features, out_features) * 0.1, need_grad=True)
        b = array(np.zeros(out_features), need_grad=True)
        self.w = Parameter(w)
        self.b = Parameter(b)

    def __call__(self, x):
        if not isinstance(x, array):
            x = array(x)
        return x @ self.w.data + self.b.data

