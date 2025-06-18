from deriv import array, exp, log, sum as dsum, mean
from deriv.Array.backend import get_backend

class SoftmaxCrossEntropy:
    def __init__(self, axis=-1):
        self.axis = axis

    def __call__(self, logits: 'array', targets: 'array') -> 'array':
        xp = get_backend()
        axis = self.axis
        batch_size = logits.shape[0]

        shift = logits.max(axis=axis, keepdims=True)
        stable_logits = logits - shift
        exps = exp(stable_logits)
        sum_exps = dsum(exps, axis=axis, keepdims=True)
        softmax = exps / sum_exps

        log_softmax = log(softmax + 1e-9)
        ce_loss = -(targets * log_softmax).sum(axis=axis)
        loss = mean(ce_loss)
        loss.parents = (logits,)
        
        def CCEBackward():
            grad_logits = (softmax.data - targets.data) / batch_size
            logits.grad += grad_logits * loss.grad
        loss._back = CCEBackward

        return loss
