from deriv import array

def grads_on(_inp:list):
    for i in _inp:
        if i.need_grad != True:
            i.need_grad = True
    return _inp
        