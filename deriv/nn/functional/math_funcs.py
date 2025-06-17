from deriv.Array.backend import get_backend
from deriv import array

def relu(x:array):
    xp = get_backend()
    return xp.maximum(x.data, 0)

def tanh(x:array):
    xp = get_backend()
    return xp.tanh(x.data)

def sigmoid(x:array):
    xp = get_backend()
    return 1/(1+xp.exp(-x.data))

def exp(x:array):
    xp = get_backend()
    return xp.exp(x.data)

def sin(x:array):
    xp = get_backend()
    return xp.sin(x.data)

def tan(x:array):
    xp = get_backend()
    return xp.tan(x.data)

def cos(x:array):
    xp = get_backend()
    return xp.cos(x.data)

