from deriv import array
import deriv
from deriv.Array.backend import get_backend


class Nami:
    def __init__(self, w_init = 0.5, a_init = 1.0, b_init = 1.5, learnable=True):
        self.w_init = array(w_init)
        self.a_init = array(a_init)
        self.b_init = array(b_init)
        if learnable:
            self.w_init, self.a_init, self.w_init = deriv.grads_on([self.w_init, self.a_init, self.w_init])

    def __call__(self, x):
        xp = get_backend()
        tanh = deriv.nn.Tanh()
        w = xp.clip(self.w_init.data, 0.0001, None)
        b = xp.clip(self.b_init.data, 0.0001, None)

        pos = tanh(x) * self.a_init
        neg = self.a_init * deriv.sin(x * w) / b

        return deriv.where(x>array(0), pos, neg)
        