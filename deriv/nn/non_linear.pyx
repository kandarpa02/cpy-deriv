# cython: boundscheck=False, wraparound=False
# distutils: language=c++

from  deriv.engine import tensor
from  deriv.utils.helpers import *
from  deriv.math.matrix_cpu import *
from  deriv.math.math_cpu import *

cdef relu_grad(object x):
    cdef int x_m, x_n, i, j
    x,_ = fix_dim(x, 0)
    x_m = len(x)
    x_n = len(x[0])
    grad = zeros_like_ct(x)
    for i in range(x_m):
        for j in range(len(x[i])):
            if x[i][j]>0:
                grad[i][j] += 1
    out = check_dim(grad)
    if get_shape(out) == (1,1):
        return out[0]
    return out


cdef class ReLU:
    def __init__(self) -> None:
        pass
    @staticmethod
    def __call__(object _obj):
        cdef object out

        if not isinstance(_obj, tensor):
            _obj, _ = fix_dim(_obj, 0)
            _obj = tensor(_obj)

        elif isinstance(_obj, tensor):
            _obj.data, _ = fix_dim(_obj.data, 0)

        else:
            raise ValueError("Object {type(_obj)} is not supported")

        out = tensor(maximum(_obj.data, 0), (_obj,), need_grad=True)

        def reluBackward():
            obj_grad = relu_grad(_obj.data)
            _obj.grad = addition(_obj.grad, multiplication(obj_grad, out.grad))

        out._back = reluBackward
        return out


cdef class Tanh:
    def __init__(self) -> None:
        pass
    @staticmethod
    def __call__(object _obj):
        cdef object out
        if not isinstance(_obj, tensor):
            _obj, _ = fix_dim(_obj, 0)
            _obj = tensor(_obj)

        elif isinstance(_obj, tensor):
            _obj.data, _ = fix_dim(_obj.data, 0)

        else:
            raise ValueError("Object {type(_obj)} is not supported")
            
        out = _obj.apply_fn(tanh_f)
        out.parents = (_obj, )

        def tanhBackward():
            def th_grad(x): 
                return 1 - x**2

            obj_grad = out.apply_fn(th_grad).data
            _obj.grad = addition(_obj.grad, multiplication(obj_grad, out.grad))

        out._back = tanhBackward
        return out
