from deriv.Array.array_object import array, unbroadcast
import numpy as np

def convert(data):
    if not isinstance(data, array):
        return array(data)
    return data

class trigo:
    @staticmethod
    def sin(obj, deg):
        obj = convert(obj)
        if deg:
            radians = np.radians(obj.data)
        else:
            radians = obj.data
            
        out = array(np.sin(radians), (obj,), need_grad=True)

        def sinBackward():
            if obj.need_grad:
                obj.grad += out.grad * np.cos(radians)

        out._back = sinBackward
        return out

    @staticmethod
    def cos(obj, deg):
        obj = convert(obj)
        if deg:
            radians = np.radians(obj.data)
        else:
            radians = obj.data
        out = array(np.cos(radians), (obj,), need_grad=True)

        def cosBackward():
            if obj.need_grad:
                obj.grad += out.grad * -np.sin(radians)

        out._back = cosBackward
        return out

class expo:
    @staticmethod
    def exp(obj):
        obj = convert(obj)
        out = array(np.exp(obj.data), (obj,), need_grad=True)

        def expBackward():
            if obj.need_grad:
                obj.grad += out.grad * out.data

        out._back = expBackward
        return out
    @staticmethod
    def log(obj):
        obj = convert(obj)
        out = array(np.log(obj.data), (obj,), need_grad=True)

        def logBackward():
            if obj.need_grad:
                obj.grad += out.grad * (1/obj.data)
        
        out._back = logBackward
        return out
    
    @staticmethod
    def log10(obj):
        obj = convert(obj)
        out = array(np.log10(obj.data), (obj,), need_grad=True)

        def log10Backward():
            if obj.need_grad:
                obj.grad += out.grad * (1/obj.data) * np.log10(np.exp(1))
        
        out._back = log10Backward
        return out

    @staticmethod
    def rootof(obj, _pow):
        obj, _pow = convert(obj), convert(1/_pow)
        out = array((obj**_pow).data, (obj, _pow), need_grad=True)

        def rootBackward():
            if obj.need_grad:
                grad_obj = _pow.data * (obj.data ** (_pow.data - 1)) * out.grad
                obj.grad += unbroadcast(grad_obj, obj.data.shape)
            
            if _pow.need_grad:
                grad_pow = out.data * np.log(obj.data) * out.grad
                _pow.grad += unbroadcast(grad_pow, _pow.data.shape)
        
        out._back = rootBackward
        return out
    
class reduct:
    @staticmethod
    def sum(obj, axis=None, keepdims=False):
        out = array(obj.data.sum(axis=axis, keepdims=keepdims), (obj,), need_grad=True)  

        def sumBackward():
            if obj.need_grad:
                grad = out.grad
                if axis is not None and not keepdims:
                    shape = list(obj.data.shape)
                    if isinstance(axis, int):
                        axis_ = [axis]
                    else:
                        axis_ = axis
                    for ax in axis_:
                        grad = np.expand_dims(grad, ax)
                obj.grad += grad * np.ones_like(obj.data)

        out._back = sumBackward
        return out

    @staticmethod
    def mean(obj, axis=None):
        out = array(obj.data.mean(axis=axis), (obj,), need_grad=True)  

        def meanBackward():
            if obj.need_grad:
                shape = obj.data.shape
                if axis is None:
                    num_elements = np.prod(shape)
                    grad = np.ones_like(obj.data) / num_elements
                    obj.grad + out.grad * grad
                else:
                    num_elements = shape[axis]
                    grad = np.ones_like(obj.data) / num_elements
                    obj.grad += out.grad * grad 

        out._back = meanBackward
        return out

    @staticmethod
    def prod(obj):
        pass
        # Will Add later

t = trigo()
e = expo()
r = reduct()

def sin(x, deg=True): return t.sin(x, deg=deg)
def cos(x, deg=True): return t.cos(x, deg=deg)
def exp(x): return e.exp(x)
def log(x): return e.log(x)
def log10(x): return e.log10(x)
def rootof(x, y): return e.rootof(x, y)
def mean(x): return r.mean(x)
def sum(x): return r.sum(x)
def prod(x): return r.prod(x)

__all__ = ['sin', 'cos', 'exp', 'log', 'log10', 'rootof', 'mean', 'sum', 'prod']