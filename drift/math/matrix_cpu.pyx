# cython: boundscheck=False, wraparound=False, cdivision=True
# distutils: language=c++

from libcpp.vector cimport vector
from libcpp cimport bool
from cython.operator cimport dereference as deref


# ==========================
# Type Conversion Utilities
# ==========================

cdef vector[float] pylist_to_vec_float(list pylist):
    cdef vector[float] vec
    vec.reserve(len(pylist))
    for item in pylist:
        vec.push_back(<float>item)
    return vec

cdef vector[vector[float]] to_vector_2d(list pylist_2d):
    cdef vector[vector[float]] result
    result.reserve(len(pylist_2d))
    for row in pylist_2d:
        result.push_back(pylist_to_vec_float(row))
    return result

cdef list to_list_2d(const vector[vector[float]] &vec):
    cdef list out = []
    for row in vec:
        out.append([x for x in row])
    return out


# ==========================
# Shape Utilities
# ==========================

cpdef tuple get_shape(list a):
    a, _ = fix_dim(a, 0)
    return (len(a), len(a[0]))

cpdef list check_dim(list a):
    """Returns a as-is or flattens the list if it's 2D with only one row or column."""
    cdef int count = 0
    for item in a:
        count += 1
        if isinstance(item, (int, float)) or (isinstance(item, list) and count > 1):
            return a
    return a[0] if isinstance(a[0], list) else a


# ==========================
# Dim Fixing and Reduction
# ==========================

def _fix_dim(object a, object b):
    if isinstance(a, (int, float)):
        a = [[a]]
    elif all(not isinstance(el, list) for el in a):
        a = [a]

    if isinstance(b, (int, float)):
        b = [[b]]
    elif all(not isinstance(el, list) for el in b):
        b = [b]

    return a, b

cpdef object fix_dim(object a, object b):
    return _fix_dim(a, b)


def _reduce_grad(object grad, tuple original_shape):
    if len(original_shape) == 0:
        return array_sum(grad, axis=None)

    grad_m, grad_n = len(grad), len(grad[0])
    orig_m, orig_n = original_shape

    if orig_m == 1 and grad_m > 1:
        grad = [[sum(grad[i][j] for i in range(grad_m)) for j in range(grad_n)]]

    if orig_n == 1 and grad_n > 1:
        grad = [[sum(row)] for row in grad]

    return grad

cpdef object reduce_grad(object grad, tuple original_shape):
    return _reduce_grad(grad, original_shape)


# ==========================
# Basic Math Operations
# ==========================

cdef list elwisemul(list a, list b):
    a, b = fix_dim(a, b)
    cdef vector[vector[float]] va = to_vector_2d(a)
    cdef vector[vector[float]] vb = to_vector_2d(b)
    cdef vector[vector[float]] result

    cdef size_t i, j
    cdef size_t rows = va.size()

    if rows != vb.size():
        raise ValueError("Shape mismatch: different row counts")

    result.resize(rows)
    for i in range(rows):
        cols = va[i].size()
        if cols != vb[i].size():
            raise ValueError("Shape mismatch at row %d" % i)
        result[i].resize(cols)
        for j in range(cols):
            result[i][j] = va[i][j] * vb[i][j]

    return to_list_2d(result)


cpdef list matmul(list a, list b):
    a, b = fix_dim(a, b)
    cdef int m = len(a), n = len(a[0]), p = len(b[0])

    cdef vector[float] a_flat, b_flat, result_flat
    cdef int i, j, k

    a_flat.reserve(m * n)
    b_flat.reserve(n * p)
    result_flat.resize(m * p)

    for i in range(m):
        for j in range(n):
            a_flat.push_back(a[i][j])
    for i in range(n):
        for j in range(p):
            b_flat.push_back(b[i][j])

    for i in range(m):
        for j in range(p):
            for k in range(n):
                result_flat[i * p + j] += a_flat[i * n + k] * b_flat[k * p + j]

    result = [[result_flat[i * p + j] for j in range(p)] for i in range(m)]
    return check_dim(result)


cpdef list transpose(list A):
    cdef int m = len(A), n = len(A[0])
    return [[A[i][j] for i in range(m)] for j in range(n)]


cpdef list expand(list mat, int target_m, int target_n):
    cdef int orig_m = len(mat), orig_n = len(mat[0])

    if orig_m == 1:
        mat = mat * target_m
    elif orig_m != target_m:
        raise ValueError("Incompatible number of rows")

    if orig_n == 1:
        mat = [[row[0]] * target_n for row in mat]
    elif orig_n != target_n:
        raise ValueError("Incompatible number of columns")

    return mat


cpdef object broadcast(object a, object b):
    a, b = fix_dim(a, b)
    cdef int a_m = len(a), a_n = len(a[0])
    cdef int b_m = len(b), b_n = len(b[0])

    cdef int out_m = max(a_m, b_m)
    cdef int out_n = max(a_n, b_n)

    a_full = expand(a, out_m, out_n)
    b_full = expand(b, out_m, out_n)

    return [[a_full[i][j] + b_full[i][j] for j in range(out_n)] for i in range(out_m)]


cpdef object addition(object a, object b):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a + b
    return broadcast(a, b)


cpdef object multiplication(object a, object b):
    cdef vector[vector[float]] vecA
    cdef int i

    if isinstance(a, (int, float)) and isinstance(b, list):
        a, b = b, a
    if isinstance(a, list) and isinstance(b, (int, float)):
        a, _ = fix_dim(a, 0)
        vecA.reserve(len(a))
        for i in range(len(a)):
            vecA.push_back(pylist_to_vec_float(a[i]))
        return [[vecA[i][j] * b for j in range(len(vecA[i]))] for i in range(len(vecA))]
    elif isinstance(a, list) and isinstance(b, list):
        return elwisemul(a, b)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a * b
    return None


# ==========================
# Array Init and Reduction
# ==========================

cpdef list ones_like_ct(list A):
    A, _ = fix_dim(A, 0)
    return [[1 for _ in range(len(A[0]))] for _ in range(len(A))]

cpdef list zeros_like_ct(list A):
    A, _ = fix_dim(A, 0)
    return [[0 for _ in range(len(A[0]))] for _ in range(len(A))]


cpdef object array_sum(object A, object axis):
    A, _ = fix_dim(A, 0)
    cdef vector[vector[float]] vecA
    for row in A:
        vecA.push_back(pylist_to_vec_float(row))

    if axis is None:
        return sum([x for row in vecA for x in row])
    elif axis == 0:
        return [sum(vecA[i][j] for i in range(len(vecA))) for j in range(len(vecA[0]))]
    elif axis == 1:
        return [sum(row) for row in vecA]
    else:
        raise ValueError("Invalid axis value. Must be None, 0, or 1.")
