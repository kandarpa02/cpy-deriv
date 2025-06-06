from deriv import array, unbroadcast
import numpy as np

class ReLU:
    """
    Rectified Linear Unit (ReLU) activation function.

    Applies the element-wise function: `ReLU(x) = max(0, x)`.

    Methods:
        __call__(_obj): Applies ReLU to the input array and sets up backward pass.
    """

    def __init__(self) -> None:
        """Initializes the ReLU activation function."""
        pass

    @staticmethod
    def __call__(_obj):
        """
        Apply the ReLU activation function to the input array.

        Args:
            _obj (array): Input tensor of type `array`.

        Returns:
            array: Output tensor after applying ReLU.

        Raises:
            ValueError: If `_obj` is not an instance of `array`.
        """
        if not isinstance(_obj, array):
            raise ValueError(f"Object of type {type(_obj)} is not supported")
        
        out = array(np.maximum(_obj.data, 0), (_obj,), need_grad=True, op="relu")

        def reluBackward():
            if _obj.need_grad:
                obj_grad = np.where(_obj.data > 0, 1.0, 0.0)
                _obj.grad += unbroadcast(obj_grad * out.grad, _obj.data.shape)

        out._back = reluBackward
        return out


class Tanh:
    """
    Hyperbolic Tangent (Tanh) activation function.

    Applies the element-wise function: `Tanh(x) = tanh(x)`, outputting values in [-1, 1].

    Methods:
        __call__(_obj): Applies Tanh to the input array and sets up backward pass.
    """

    def __init__(self) -> None:
        """Initializes the Tanh activation function."""
        pass

    @staticmethod
    def __call__(_obj):
        """
        Apply the Tanh activation function to the input array.

        Args:
            _obj (array): Input tensor of type `array`.

        Returns:
            array: Output tensor after applying Tanh.

        Raises:
            ValueError: If `_obj` is not an instance of `array`.
        """
        if not isinstance(_obj, array):
            raise ValueError(f"Object of type {type(_obj)} is not supported")

        out = array(np.tanh(_obj.data), (_obj,), need_grad=True, op="tanh")

        def tanhBackward():
            if _obj.need_grad:
                grad_val = 1.0 - np.tanh(_obj.data) ** 2
                _obj.grad += unbroadcast(grad_val * out.grad, _obj.data.shape)

        out._back = tanhBackward
        return out
