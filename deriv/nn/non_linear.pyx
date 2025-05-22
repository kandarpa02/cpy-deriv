# cython: boundscheck=False, wraparound=False
# distutils: language=c++

from deriv.engine import *
import numpy as np

cdef class ReLU:
    def __init__(self) -> None:
        pass

    @staticmethod
    def __call__(object _obj):
        cdef object out

        if not isinstance(_obj, array):
            raise ValueError(f"Object of type {type(_obj)} is not supported")
        
        out = array(np.maximum(_obj.data, 0), (_obj,), need_grad=True)

        def reluBackward():
            if _obj.need_grad:
                obj_grad = np.where(_obj.data > 0, 1.0, 0.0)
                _obj.grad += unbroadcast(obj_grad * out.grad, _obj.data.shape)

        out._back = reluBackward
        return out


cdef class Tanh:
    def __init__(self) -> None:
        pass

    @staticmethod
    def __call__(object _obj):
        cdef object out

        if not isinstance(_obj, array):
            raise ValueError(f"Object of type {type(_obj)} is not supported")

        out = array(np.tanh(_obj.data), (_obj,), need_grad=True)

        def tanhBackward():
            if _obj.need_grad:
                grad_val = 1.0 - np.tanh(_obj.data) ** 2
                _obj.grad += unbroadcast(grad_val * out.grad, _obj.data.shape)

        out._back = tanhBackward
        return out
