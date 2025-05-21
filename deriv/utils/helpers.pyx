# cython: boundscheck=False, wraparound=False
# distutils: language=c++

from  deriv.engine import *
from  deriv.math.matrix_cpu import *
from  deriv.utils.util_excp import *

cpdef object maximum(object a, object b):
    """
    Element-wise maximum between two tensors (broadcasted if necessary).
    """
    cdef int a_m, a_n, b_m, b_n, out_m, out_n, i
    cdef list result = []

    if isinstance(a, tensor):
        a = a.data
    if isinstance(b, tensor):
        b = b.data

    a, b = fix_dim(a, b)
    
    a_m, a_n = len(a), len(a[0])
    b_m, b_n = len(b), len(b[0])

    out_m, out_n = max(a_m, b_m), max(a_n, b_n)

    a_full = expand(a, out_m, out_n)
    b_full = expand(b, out_m, out_n)

    for i in range(out_m):
        result.append([max(j, k) for j, k in zip(a_full[i], b_full[i])])

    if not isinstance(result[0][0], list) and get_shape(result) == (1, 1):
        return result[0][0]
    else:
        return check_dim(result)


cpdef object _argmax(object a, object axis):
    """
    Core argmax function supporting flattened and axis=0.
    """
    cdef int i, j, k, idx
    cdef float val, max_val
    cdef list result, pair, result_idx
    if isinstance(a, tensor):
        a = a.data
    a, _ = fix_dim(a, 0)

    if axis is None:
        # Flatten
        result = []
        for row in a:
            result.extend(row)

        max_val = result[0]
        idx = 0
        for i, val in enumerate(result):
            if val > max_val:
                max_val = val
                idx = i
        return tensor(idx)

    elif axis == 0:
        result_idx = []
        for i in range(len(a[0])):  # column-wise
            pair = [row[i] for row in a]
            max_val = pair[0]
            idx = 0
            for k, val in enumerate(pair):
                if val > max_val:
                    max_val = val
                    idx = k
            result_idx.append(idx)
        return tensor(result_idx)

    elif axis == 1:
        result_idx = []
        for i in range(len(a)):
            max_val = a[i][0]
            idx = 0
            for k, val in enumerate(a[i]):
                if val > max_val:
                    max_val = val
                    idx = k
            result_idx.append(idx)
        return tensor(result_idx)
    else:
        raise AxisError(axis)


def argmax(a, axis=None):
    """
    Parameters
    ----------
    a : array_like
        Input array.
    axis : int, optional
        Axis along which to compute. If None, the array is flattened.

    Returns
    -------
    int or tensor
        Index/indices of the maximum values.

    Examples
    --------
    >>> argmax([[1, 2, 4], [5, 6, 6]])
    tensor(4)
    >>> argmax([[1, 2, 4], [5, 6, 6]], axis=0)
    tensor([1, 1, 1])
    """
    return _argmax(a, axis)
