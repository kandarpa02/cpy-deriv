from deriv import array
import deriv
from deriv.Array.backend import get_backend

class SoftmaxCrossEntropy:
    def __init__(self, axis=-1):
        self.axis = axis

    def __call__(self, logits: 'array', targets: 'array') -> 'array':
        """
        Compute softmax cross-entropy loss with automatic gradient calculation.
        
        Args:
            logits: Unnormalized log probabilities (array)
            targets: Ground truth labels (array, one-hot encoded)
        
        Returns:
            Loss value (scalar array)
        """
        xp = get_backend()
        
        shift = logits.max(axis=self.axis, keepdims=True)
        stable_logits = logits - shift
        
        exps = deriv.exp(stable_logits)
        sum_exps = exps.sum(axis=self.axis, keepdims=True)
        softmax = exps / sum_exps
        
        batch_size = logits.data.shape[0]
        log_softmax = deriv.log((softmax + 1e-9))
        ce_loss = -(targets * log_softmax).sum(axis=self.axis)
        loss = ce_loss.mean()
        
        loss._back = self._build_backward_fn(
            logits, targets, softmax, batch_size, loss, xp
        )
        
        return loss

    def _build_backward_fn(self, logits, targets, softmax, batch_size, loss, xp):
        def backward():
            grad_logits = (softmax.data - targets.data) / batch_size
            
            if logits.need_grad:
                logits.grad += grad_logits * loss.grad
        
        return backward