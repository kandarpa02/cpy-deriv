from deriv import array
from deriv import exp, log, sum as dsum, mean
from deriv.Array.backend import get_backend

class SoftmaxCrossEntropy:
    def __init__(self, axis=-1):
        self.axis = axis

    def __call__(self, logits: 'array', targets: 'array') -> 'array':
        xp = get_backend()
        shift = logits.max(axis=self.axis, keepdims=True)
        stable_logits = logits - shift

        exps = exp(stable_logits)
        softmax = exps / dsum(exps, axis=self.axis, keepdims=True)

        log_softmax = log(softmax + 1e-9)
        ce = dsum(targets * log_softmax, axis=self.axis)
        return -mean(ce)
