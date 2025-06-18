from deriv import array
from deriv.Array.numerics import OPERATIONS
import codegen

class compact:
    def __init__(self, func):
        self.func = func
        self.f2, self.code = codegen.compact(self.func, debug=True)
    
    def apply(self, *args):
        return self.f2(*args)

