from deriv.Array.backend import get_backend
from deriv import array

def addt(a, b): return a + b
def mult(a, b): return a * b
def subt(a, b): return a - b
def div(a, b): return a / b
def powr(a, b): return a ** b
def matmul(a, b): return a @ b

def relu(a): 
    xp = get_backend()
    return array(xp.maximum(a.data, 0))

def tanh(a): 
    xp = get_backend()
    return array(xp.tanh(a.data))

def sigmoid(a): 
    xp = get_backend()
    return array(1/(1+xp.exp(-a.data)))

OPERATIONS = {

    '+': addt,
    '*': mult,
    '-': subt,
    '/': div,
    '**': powr,
    '@': matmul,
    'relu': relu,
    'tanh': tanh,
    'sigmoid': sigmoid

}
