from deriv.Array.array_object import array
import numpy as np

class trigo:
    @staticmethod
    def sin(obj):
        out = array(np.sin(obj.data), (obj,), need_grad=True)

        def sinBackward():
            if obj.need_grad:
                obj.grad += out.grad * np.cos(obj.data)

        out._back = sinBackward
        return out

    @staticmethod
    def cos(obj):
        out = array(np.cos(obj.data), (obj,), need_grad=True)

        def cosBackward():
            if obj.need_grad:
                obj.grad += out.grad * -np.sin(obj.data)

        out._back = cosBackward
        return out

t = trigo()

def sin(x): return t.sin(x)
def cos(x): return t.cos(x)


__all__ = ['sin', 'cos']