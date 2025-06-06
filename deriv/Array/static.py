from deriv import array
from deriv.Array.numerics import OPERATIONS

def is_unary(op):
    func = OPERATIONS[op]
    try:
        return func.__code__.co_argcount == 1
    except AttributeError:
        return False

def fuse(vars, ops):
    def get_func_name(op): 
        func = OPERATIONS[op]
        return func.__name__ if callable(func) else str(func)

    vars = vars[:] 
    i = 0 
    j = 1 

    expr = f"{vars[0]}"
    while i < len(ops):
        op = ops[i]
        func = get_func_name(op)
        if is_unary(op):
            expr = f"{func}({expr})"
        else:
            expr = f"{func}({expr}, {vars[j]})"
            j += 1
        i += 1

    return expr

def XLR(fn_out:array, debug=False):
    graph = fn_out.topo()

    nodes = []
    ops = []
    for node in graph:
        if node.op == '':
            nodes.append(node)
        else:
            ops.append(node.op)

    variables = {}
    for i, node in enumerate(nodes):
        variables[node] = f"x_{i}"

    _vars = list(variables.values())

    opmap = fuse(_vars, ops)

    args = ', '.join(_vars)
    code = f"def fused({args}):\n"
    code += f"    return {opmap}\n"

    scope = {func.__name__: func for func in OPERATIONS.values()}
    exec(code, scope)

    if debug:
        return scope['fused'], code
    else:
        return scope['fused']