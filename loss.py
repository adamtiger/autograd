from tensor import Tensor
import numpy as np

class Loss:
    def __call__(self, *params):
        loss = self._forward(*params)
        self._backward(*params, loss)
        return loss
    
    def _forward(self, *params):
        raise NotImplementedError("Loss forward requires implementation")
    
    def _backward(self, *params, outputs):
        raise NotImplementedError("Loss backward requires implementation")


class MSE(Loss):
    def _forward(self, y_real: Tensor, y_predicted: Tensor) -> Tensor:
        y = Tensor([1], y_real.dtype, y_predicted.requires_grad)
        y_diff = (y_real.data - y_predicted.data)
        y.data = np.array([np.mean(y_diff * y_diff)])
        return y
    
    def _backward(self, y_real: Tensor, y_predicted: Tensor, y: Tensor):
        n = y_real.data.size
        rn = 1.0 / float(n)
        y.add_parent(y_predicted, lambda tot_grad: tot_grad * (y_predicted.data - y_real.data) * 2.0 * rn)
        y.total_grad = 1.0  # source
