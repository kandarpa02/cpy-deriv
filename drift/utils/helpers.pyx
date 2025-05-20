# cython: boundscheck=False, wraparound=False
# distutils: language=c++

from drift.engine import *
from drift.math.matrix_cpu import *

cpdef object maximum(object a, object b):
    cdef int a_m, a_n, b_m, b_n, out_m, out_n, i
    cdef list result, out

    a, b = fix_dim(a, b)
    a_m = len(a)
    a_n = len(a[0])
    b_m = len(b)
    b_n = len(b[0])

    out_m = max(a_m, b_m)
    out_n = max(a_n, b_n)

    result = [[] for _ in range(out_m)]

    a_full = expand(a, out_m, out_n)
    b_full = expand(b, out_m, out_n)

    for i in range(out_m):
        for j, k in zip(a_full[i], b_full[i]):
            result[i].append(max(j,k))
    out = check_dim(result)

    if get_shape(out) == (1,1):
        return out[0]

    return out

cpdef object _argmax(object a, object axis):

    if isinstance(a, (int, tuple, list)):
        a = tensor(a)
    cdef int ixd, i
    a, _ = fix_dim(a.data, 0)
    cdef list result, result_idx, pair
    cdef float val, max_val
    if axis is None:
        result = []
        for i in range(len(a)):
            for j in a[i]:
                result.append(j)

        a_flat = result

        max_val, idx = a_flat[0], 0
        for i, val in enumerate(a_flat):
            if val>max_val:
                max_val, idx = val, i
        return tensor(idx)

    elif axis==0:
        result_idx = []

        for i in range(len(a[0])):
            pair = []
            for j in a:
                pair.append(j[i])
                max_val, idx = pair[0], 0
                for k, val in enumerate(pair):
                    if val>max_val:
                        max_val, idx = val, k
            result_idx.append(idx)
        return tensor(result_idx)


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
    int or ndarray
        Index (or indices) of the maximum values.

    Examples
    --------
    >>> import drift as ft
    >>> x = [[1, 2, 4], [5, 6, 6]]
    >>> ft.argmax(x)
    tensor(4)
    >>> ft.argmax(x, axis=0)
    tensor([1, 1, 1])
    """

    return _argmax(a, axis)



        
    