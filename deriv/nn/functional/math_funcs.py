from deriv.Array.backend import get_backend
from deriv import array
from codegen.backend.core.symbolic.tracer_object import Tracer

def relu(x:array):
    xp = get_backend()
    if isinstance(x, Tracer):
        return Tracer(f"relu({x})", op = 'relu', parents = (x, ))
    return xp.maximum(x.data, 0)

def tanh(x:array):
    xp = get_backend()
    if isinstance(x, Tracer):
        return Tracer(f"tanh({x})", op = 'tanh', parents = (x, ))
    return xp.tanh(x.data)

def sigmoid(x:array):
    xp = get_backend()
    if isinstance(x, Tracer):
        return Tracer(f"sigmoid({x})", op = 'sigmoid', parents = (x, ))
    return 1/(1+xp.exp(-x.data))

def exp(x:array):
    xp = get_backend()
    if isinstance(x, Tracer):
        return Tracer(f"exp({x})", op = 'exp', parents = (x, ))
    return xp.exp(x.data)

def sin(x:array):
    xp = get_backend()
    if isinstance(x, Tracer):
        return Tracer(f"sin({x})", op = 'sin', parents = (x, ))
    return xp.sin(x.data)

def tan(x:array):
    xp = get_backend()
    if isinstance(x, Tracer):
        return Tracer(f"tan({x})", op = 'tan', parents = (x, ))
    return xp.tan(x.data)

def cos(x:array):
    xp = get_backend()
    if isinstance(x, Tracer):
        return Tracer(f"cos({x})", op = 'cos', parents = (x, ))
    return xp.cos(x.data)
