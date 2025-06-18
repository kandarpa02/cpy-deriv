from deriv.Array.array_object import array, unbroadcast
from deriv.Array.backend import get_backend


def convert(data):
    """
    Converts input to an `array` instance if it is not already.

    Args:
        data: A numeric value, NumPy array, or already an `array` instance.

    Returns:
        `array` instance wrapping the input.
    """
    if not isinstance(data, array):
        return array(data)
    return data

class trigo:
    """
    Trigonometric functions with support for autograd.
    """

    @staticmethod
    def sin(obj, deg):
        xp = get_backend()
        """
        Compute the sine of the input.

        Args:
            obj: Input `array` or compatible type.
            deg (bool): If True, interpret input as degrees. Otherwise, radians.

        Returns:
            `array`: Result of sin operation with autograd support.
        """
        obj = convert(obj)
        radians = xp.radians(obj.data) if deg else obj.data
        out = array(xp.sin(radians), (obj,), need_grad=True, op='sin')

        def sinBackward():
            if obj.need_grad:
                obj.grad += out.grad * xp.cos(radians)

        out._back = sinBackward
        return out

    @staticmethod
    def cos(obj, deg):
        xp = get_backend()
        """
        Compute the cosine of the input.

        Args:
            obj: Input `array` or compatible type.
            deg (bool): If True, interpret input as degrees. Otherwise, radians.

        Returns:
            `array`: Result of cos operation with autograd support.
        """
        obj = convert(obj)
        radians = xp.radians(obj.data) if deg else obj.data
        out = array(xp.cos(radians), (obj,), need_grad=True, op='cos')

        def cosBackward():
            if obj.need_grad:
                obj.grad += out.grad * -xp.sin(radians)

        out._back = cosBackward
        return out


class expo:
    """
    Exponential and logarithmic functions with autograd support.
    """

    @staticmethod
    def exp(obj):
        xp = get_backend()
        """
        Compute the exponential of the input.

        Args:
            obj: Input `array` or compatible type.

        Returns:
            `array`: Result of exp operation with autograd support.
        """
        obj = convert(obj)
        out = array(xp.exp(obj.data), (obj,), need_grad=True, op='exp')

        def expBackward():
            if obj.need_grad:
                obj.grad += out.grad * out.data

        out._back = expBackward
        return out

    @staticmethod
    def log(obj):
        xp = get_backend()
        """
        Compute the natural logarithm of the input.

        Args:
            obj: Input `array` or compatible type.

        Returns:
            `array`: Result of log operation with autograd support.
        """
        obj = convert(obj)
        out = array(xp.log(obj.data), (obj,), need_grad=True, op='log')

        def logBackward():
            if obj.need_grad:
                obj.grad += out.grad * (1 / obj.data)

        out._back = logBackward
        return out

    @staticmethod
    def log10(obj):
        xp = get_backend()
        """
        Compute the base-10 logarithm of the input.

        Args:
            obj: Input `array` or compatible type.

        Returns:
            `array`: Result of log10 operation with autograd support.
        """
        obj = convert(obj)
        out = array(xp.log10(obj.data), (obj,), need_grad=True, op='log10')

        def log10Backward():
            if obj.need_grad:
                obj.grad += out.grad * (1 / obj.data) * xp.log10(xp.exp(1))

        out._back = log10Backward
        return out

    @staticmethod
    def rootof(obj, _pow):
        xp = get_backend()
        """
        Compute the nth root of an input as `obj^(1/_pow)`.

        Args:
            obj: Base input `array` or compatible type.
            _pow: Root power (e.g., 2 for square root).

        Returns:
            `array`: Result of root operation with autograd support.
        """
        obj, _pow = convert(obj), convert(1 / _pow)
        out = array((obj ** _pow).data, (obj, _pow), need_grad=True, op='root')

        def rootBackward():
            if obj.need_grad:
                grad_obj = _pow.data * (obj.data ** (_pow.data - 1)) * out.grad
                obj.grad += unbroadcast(grad_obj, obj.data.shape)

            if _pow.need_grad:
                grad_pow = out.data * xp.log(obj.data) * out.grad
                _pow.grad += unbroadcast(grad_pow, _pow.data.shape)

        out._back = rootBackward
        return out


class reduct:
    """
    Reduction operations (sum, mean) with autograd support.
    """

    @staticmethod
    def sum(obj, axis=None, keepdims=False):
        xp = get_backend()
        """
        Compute the sum along specified axis.

        Args:
            obj: Input `array`.
            axis (int or tuple of ints): Axis or axes to reduce.
            keepdims (bool): If True, retains reduced dimensions.

        Returns:
            `array`: Result of sum operation with autograd support.
        """
        out = array(obj.data.sum(axis=axis, keepdims=keepdims), (obj,), need_grad=True, op='sum')

        def sumBackward():
            if obj.need_grad:
                grad = out.grad
                if axis is not None and not keepdims:
                    shape = list(obj.data.shape)
                    axis_ = [axis] if isinstance(axis, int) else axis
                    for ax in axis_:
                        grad = xp.expand_dims(grad, ax)
                obj.grad += grad * xp.ones_like(obj.data)

        out._back = sumBackward
        return out

    @staticmethod
    def mean(obj, axis=None):
        xp = get_backend()
        """
        Compute the mean along specified axis.

        Args:
            obj: Input `array`.
            axis (int, optional): Axis to compute mean over.

        Returns:
            `array`: Result of mean operation with autograd support.
        """
        out = array(obj.data.mean(axis=axis), (obj,), need_grad=True, op='mean')

        def meanBackward():
            if obj.need_grad:
                shape = obj.data.shape
                if axis is None:
                    num_elements = xp.prod(shape)
                    grad = xp.ones_like(obj.data) / num_elements
                    obj.grad += out.grad * grad
                else:
                    num_elements = shape[axis]
                    grad = xp.ones_like(obj.data) / num_elements
                    obj.grad += out.grad * grad

        out._back = meanBackward
        return out

    @staticmethod
    def prod(obj):
        xp = get_backend()
        """
        Placeholder for product reduction function.

        Args:
            obj: Input `array`.

        Returns:
            Not implemented.
        """
        pass  # Will add later


# Functional API aliases for convenience
t = trigo()
e = expo()
r = reduct()

def sin(x, deg=False): return t.sin(x, deg=deg)
def cos(x, deg=False): return t.cos(x, deg=deg)
def exp(x): return e.exp(x)
def log(x): return e.log(x)
def log10(x): return e.log10(x)
def rootof(x, y): return e.rootof(x, y)
def mean(x, axis=None): return r.mean(x, axis=axis)
def sum(x, axis=None, keepdims=False): return r.sum(x, axis=axis, keepdims=keepdims)
def prod(x): return r.prod(x)

__all__ = ['sin', 'cos', 'exp', 'log', 'log10', 'rootof', 'mean', 'sum', 'prod']
