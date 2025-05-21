# cython: boundscheck=False, wraparound=False
# distutils: language=c++
from libcpp.vector cimport vector
from libcpp cimport bool


cdef vector[float] pylist_to_vec_float(list pylist):
    cdef vector[float] vec
    cdef Py_ssize_t i
    for i in range(len(pylist)):
        vec.push_back(float(pylist[i]))
    return vec

cdef vector[vector[float]] to_vector_2d(list lst):
    """Convert Python list of lists to std::vector<std::vector[float]]"""
    cdef vector[vector[float]] result
    cdef vector[float] row
    cdef list inner
    cdef float val
    for inner in lst:
        row = vector[float]()
        for val in inner:
            row.push_back(<float>val)
        result.push_back(row)
    return result

cdef list to_list_2d(const vector[vector[float]] &vec):
    """Convert std::vector<std::vector[float]] back to list of lists"""
    cdef list outer = []
    cdef list inner
    cdef size_t i, j
    for i in range(vec.size()):
        inner = []
        for j in range(vec[i].size()):
            inner.append(vec[i][j])
        outer.append(inner)
    return outer


# Helper func 1 
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
    return _fix_dim(a,b)



# Helper func 2
def _reduce_grad(object grad, tuple original_shape):
    # Handle scalar case (original_shape == ())
    if len(original_shape) == 0:
        return array_sum(grad, axis=None)  # return a scalar value

    cdef int grad_m = len(grad)
    cdef int grad_n = len(grad[0])
    cdef int orig_m = original_shape[0]
    cdef int orig_n = original_shape[1]
    cdef int i, j

    # Reduce rows if originally single row (e.g., (1, 2))
    if orig_m == 1 and grad_m > 1:
        grad = [[sum(grad[i][j] for i in range(grad_m)) for j in range(grad_n)]]

    # Reduce columns if originally single column (e.g., (2, 1))
    if orig_n == 1 and grad_n > 1:
        grad = [[sum(row) ] for row in grad]

    return grad

cpdef object reduce_grad(object grad, tuple original_shape):
    return _reduce_grad(grad, original_shape)


# Helper func 3
cpdef tuple get_shape(list a):
    if isinstance(a, (int, float)):
        raise ValueError("Only vector object can have a shape")
    a, _ = fix_dim(a, 0)
    a_m, a_n = len(a), len(a[0])
    return (a_m, a_n)


# Helper func 4
cpdef list check_dim(list a):
    cdef list _
    cdef int count
    cdef object i
    _ = []
    count = 0
    for i in a:
        count += 1
        if isinstance(i, (int, float)):
            _ = a
        elif isinstance(i, list) and count>1:
            _ = a
        else:
            _ = i
    return _


cdef elwisemul(list a, list b):                                                         # Elementwise multiplication
    a, b = fix_dim(a, b)                    
    cdef vector[vector[float]] va = to_vector_2d(a)
    cdef vector[vector[float]] vb = to_vector_2d(b)
    cdef vector[vector[float]] result

    cdef size_t i, j
    cdef size_t rows = va.size()
    cdef size_t cols

    # Enforce strict shape match
    if rows != vb.size():
        raise ValueError(f"Shape mismatch: row count {rows} vs {vb.size()}")

    for i in range(rows):
        if va[i].size() != vb[i].size():
            raise ValueError(f"Shape mismatch at row {i}: "
                             f"{va[i].size()} vs {vb[i].size()}")

    result.resize(rows)

    for i in range(rows):
        cols = va[i].size()
        result[i].resize(cols)
        for j in range(cols):
            result[i][j] = va[i][j] * vb[i][j]

    return to_list_2d(result)


cpdef list matmul(list a, list b):                                                            # MatMul
    a, b = fix_dim(a, b)
    cdef int m = len(a)
    cdef int n = len(a[0])
    cdef int p = len(b[0])
    
    cdef vector[float] a_flat
    cdef vector[float] b_flat
    cdef vector[float] result_flat
    cdef int i, j, k

    # Flatten A (row-major)
    for i in range(m):
        for j in range(n):
            a_flat.push_back(a[i][j])

    # Flatten B (row-major)
    for i in range(n):
        for j in range(p):
            b_flat.push_back(b[i][j])

    result_flat.resize(m * p)

    # MatMul
    for i in range(m):
        for j in range(p):
            for k in range(n):
                result_flat[i * p + j] += a_flat[i * n + k] * b_flat[k * p + j]

    # Build final 2D result
    result = []
    for i in range(m):
        row = []
        for j in range(p):
            row.append(result_flat[i * p + j])
        result.append(row)

    return check_dim(result)



cpdef list transpose(list A):
    A, _ = fix_dim(A, 0)

    cdef int m = len(A)
    cdef int n = len(A[0])

    cdef vector[vector[float]] vecA

    cdef int i

    for i in range(m):
        vecA.push_back(pylist_to_vec_float(A[i]))

    cdef list t = [[0 for _ in range(m)] for _ in range(n)]

    cdef int k, j

    for k in range(m):
        for j in range(n):
            t[j][k] = A[k][j]

    return t


cpdef list expand(mat, int target_m, int target_n):
    mat, _ = fix_dim(mat, 0)
    
    cdef int orig_m = len(mat)
    cdef int orig_n = len(mat[0])

    if orig_m == 1:
        mat = mat * target_m
    elif orig_m != target_m:
        raise ValueError("Incompatible number of rows")

    if orig_n == 1:
        mat = [[row[0]] * target_n for row in mat]
    elif orig_n != target_n:
        raise ValueError("Incompatible number of columns")

    return mat


cpdef object broadcast(object a, object b):                                                           # Broadcast
    a, b = fix_dim(a, b)

    cdef int a_m = len(a)
    cdef int a_n = len(a[0])
    cdef int b_m = len(b)
    cdef int b_n = len(b[0])

    cdef int out_m = max(a_m, b_m)
    cdef int out_n = max(a_n, b_n)

    a_full = expand(a, out_m, out_n)
    b_full = expand(b, out_m, out_n)

    cdef int i, j
    cdef list result 

    result = []
    for i in range(out_m):
        row = []
        for j in range(out_n):
            row.append(a_full[i][j] + b_full[i][j])
        result.append(row)

    return result



cpdef object addition(object a, object b):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a + b
    elif isinstance(a, list) or isinstance(b, list):
        return broadcast(a, b)


cpdef array_sum(object A, object axis):                                                                      # For implementing .sum()
    A, _ = fix_dim(A, 0)
    cdef vector[vector[float]] vecA
    cdef object result, i, j, out
    cdef float elsum
    
    for i in range(len(A)):
        vecA.push_back(pylist_to_vec_float(A[i]))

    if axis is None:
        result = 0
        for i in range(len(A)):
            for j in vecA[i]:
                result+= j

    elif axis==0:
        result = [0 for _ in A[0]]
        for v in vecA:
            for j in range(len(v)):
                result[j] += v[j]

    elif axis==1:
        result = []
        for i in range(len(A)):
            elsum = 0
            for j in range(len(A[i])):
                elsum += vecA[i][j]
            result.append(elsum)
            
    return result


cpdef object multiplication(object a, object b):                                                      # For multiplication
    cdef vector[vector[float]] vecA
    cdef object result
    cdef int i

    if isinstance(a, (int, float)) and isinstance(b, list):
        a, b = b, a
    if isinstance(a, list) and isinstance(b, (int, float)):
        # Convert a (list of lists) to vecA
        a, _ = fix_dim(a, 0)
        vecA = vector[vector[float]]()
        for i in range(len(a)):
            vecA.push_back(pylist_to_vec_float(a[i]))

        # Element-wise scalar multiplication
        result = []
        for i in range(len(a)):
            row = [vecA[i][j] * b for j in range(len(vecA[i]))]
            result.append(row)

    elif isinstance(a, list) and isinstance(b, list):
        result = elwisemul(a, b)

    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        result = a * b

    else:
        result = None

    return result


cpdef list ones_like_ct(list A):
    A, _ = fix_dim(A, 0)
    cdef int m, n
    m = len(A)
    n = len(A[0])

    return [[1 for _ in range(n)] for _ in range(m)]

cpdef list zeros_like_ct(list A):
    A, _ = fix_dim(A, 0)
    cdef int m, n
    m = len(A)
    n = len(A[0])

    return [[0 for _ in range(n)] for _ in range(m)]
