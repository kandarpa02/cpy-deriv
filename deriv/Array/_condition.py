from deriv.Array.array_object import array
from deriv.Array.backend import get_backend


def where(_statement, _do_data, _otherwise_data):
    xp = get_backend()
    """
    **deriv.where(_statement, _do_data, _otherwise_data)**

    Okay the where statement works like this:
    it takes `_statement`, `_do_data`, `_otherwise_data` as arguments
    it reads the `_statement`, if `True` follows the command in `_do_data`
    otherise do this `_otherwise_data`
    for example:
    
    >>> import deriv
    >>> x = 5
    >>> y = 2
    >>> deriv.where(x>y, x*y, x+y)
    10
    >>> x = 2
    >>> y = 5
    >>> deriv.where(x>y, x*y, x+y)
    7
    """
    out_data = xp.where(_statement.data, _do_data.data, _otherwise_data.data)
    need_grad = _do_data.need_grad or _otherwise_data.need_grad
    out = array(out_data, need_grad=need_grad)

    def whereBackward():
        if _do_data.need_grad:
            grad = xp.where(_statement.data, out.grad, 0.0)
            _do_data.grad += grad
            if hasattr(_do_data, '_back') and _do_data._back:
                _do_data._back()
        if _otherwise_data.need_grad:
            grad = xp.where(_statement.data, 0.0, out.grad)
            _otherwise_data.grad += grad
            if hasattr(_otherwise_data, '_back') and _otherwise_data._back:
                _otherwise_data._back()

    out._back = whereBackward
    return out
    