"""
    This module implements the operators
    to build graphs.
    Each operator has a derivative for backward propagation.

    tg: total gradient of y (the output of the operator)
"""

from typing import Tuple
from scipy import special as sc
from torch.nn import functional as F
import numpy as np
import torch

from tensor import Tensor


class Op:
    def __call__(self, *params):
        outputs = self._forward(*params)
        self._backward(*params, outputs)  # registers the backward related info
        return outputs
    
    def _forward(self, *params):
        raise NotImplementedError("Op forward requires implementation")
    
    def _backward(self, *params, outputs):
        raise NotImplementedError("Op backward requires implementation")


class PointwiseAdd(Op):
    def _forward(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        y = Tensor(lhs.shape, lhs.dtype, lhs.requires_grad or rhs.requires_grad)
        y.data = lhs.data + rhs.data
        return y
    
    def _backward(self, lhs: Tensor, rhs: Tensor, y: Tensor):
        y.add_parent(lhs, lambda tg : tg * np.ones_like(lhs.data))
        y.add_parent(rhs, lambda tg : tg * np.ones_like(rhs.data))

class PointwiseSub(Op):
    def _forward(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        y = Tensor(lhs.shape, lhs.dtype, lhs.requires_grad or rhs.requires_grad)
        y.data = lhs.data - rhs.data
        return y
    
    def _backward(self, lhs: Tensor, rhs: Tensor, y: Tensor):
        y.add_parent(lhs, lambda tg : tg * np.ones_like(lhs.data))
        y.add_parent(rhs, lambda tg : tg * (-np.ones_like(rhs.data)))

class PointwiseMul(Op):
    def _forward(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        y = Tensor(lhs.shape, lhs.dtype, lhs.requires_grad or rhs.requires_grad)
        y.data = lhs.data * rhs.data
        return y
    
    def _backward(self, lhs: Tensor, rhs: Tensor, y: Tensor):
        y.add_parent(lhs, lambda tg : tg * rhs.data)
        y.add_parent(rhs, lambda tg : tg * lhs.data)

class PointwiseDiv(Op):
    def _forward(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        y = Tensor(lhs.shape, lhs.dtype, lhs.requires_grad or rhs.requires_grad)
        y.data = lhs.data / rhs.data
        return y
    
    def _backward(self, lhs: Tensor, rhs: Tensor, y: Tensor):
        y.add_parent(lhs, lambda tg : tg * 1.0 / rhs.data)
        y.add_parent(rhs, lambda tg : tg * (-lhs.data / (rhs.data * rhs.data)))

class Sinus(Op):
    def _forward(self, x: Tensor) -> Tensor:
        y = Tensor(x.shape, x.dtype, x.requires_grad)
        y.data = np.sin(x.data)
        return y
    
    def _backward(self, x: Tensor, y: Tensor):
        y.add_parent(x, lambda tg: tg * np.cos(x.data))

class Cosinus(Op):
    def _forward(self, x: Tensor) -> Tensor:
        y = Tensor(x.shape, x.dtype, x.requires_grad)
        y.data = np.cos(x.data)
        return y
    
    def _backward(self, x: Tensor, y: Tensor):
        y.add_parent(x, lambda tg: tg * (-np.sin(x.data)))

class Exp(Op):
    def _forward(self, x: Tensor) -> Tensor:
        y = Tensor(x.shape, x.dtype, x.requires_grad)
        y.data = np.exp(x.data)
        return y
    
    def _backward(self, x: Tensor, y: Tensor):
        y.add_parent(x, lambda tg: tg * np.exp(x.data))

class ReLU(Op):
    def _forward(self, x: Tensor) -> Tensor:
        y = Tensor(x.shape, x.dtype, x.requires_grad)
        y.data = np.maximum(x.data, 0)
        return y
    
    def _backward(self, x: Tensor, y: Tensor):
        y.add_parent(x, lambda tg: tg * np.where(x.data > 0, 1, 0))

class GeLU(Op):
    def _forward(self, x: Tensor) -> Tensor:
        y = Tensor(x.shape, x.dtype, x.requires_grad)
        y.data = 0.5 * x.data * (1.0 + sc.erf(x.data / np.sqrt(2.0)))
        return y
    
    def _backward(self, x: Tensor, y: Tensor):
        def bwd(t):
            # erf is defined by an integral
            # therefore the derivative is the function to be integrated
            # (see explanation for details)
            t1 = 0.5 * (1.0 + sc.erf(t / np.sqrt(2.0)))
            t2 = 0.5 * t * (np.sqrt(2.0 / np.pi) * np.exp(-(t * t) / 2.0))
            return t1 + t2
        y.add_parent(x, lambda tg: tg * bwd(x.data))

class Sigmoid(Op):
    def _sgm(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    def _forward(self, x: Tensor) -> Tensor:
        y = Tensor(x.shape, x.dtype, x.requires_grad)
        y.data = self._sgm(x.data)
        return y
    
    def _backward(self, x: Tensor, y: Tensor):
        y.add_parent(x, lambda tg: tg * self._sgm(x.data) * (1.0 - self._sgm(x.data)))

class Reshape(Op):
    def __init__(self, new_shape: Tuple[int]):
        super().__init__()
        self.new_shape = new_shape

    def _forward(self, x: Tensor):
        y = Tensor(self.new_shape, x.dtype, x.requires_grad)
        y.data = np.reshape(x.data, self.new_shape)
        return y
    
    def _backward(self, x: Tensor, y: Tensor):
        # reshape does not change the number of elements
        # only reorders them
        y.add_parent(x, lambda tg: np.reshape(tg, x.shape))

class Transpose(Op):
    def __init__(self, permute: Tuple[int]):
        super().__init__()
        self.permute = permute
    
    def _forward(self, x: Tensor):
        xt = np.transpose(x.data, self.permute)
        y = Tensor(xt.shape, x.dtype, x.requires_grad)
        y.data = xt
        return y
    
    def _backward(self, x: Tensor, y: Tensor):
        # - transpose only reorders the elements in
        # the original tensor
        # - transpose can be reordered with another
        # transpose: x = Tr(Tr(x, perm), rev_perm)
        # - the gradients just go through the transpose
        # only the reordering is required
        idcs = list(range(len(self.permute)))
        rev_permute = [0] * len(self.permute)  # permute the axis to the original order
        for ix, px in zip(idcs, self.permute):
            rev_permute[px] = ix
        y.add_parent(x, lambda tg: np.transpose(tg, rev_permute))

class Softmax(Op):
    def __init__(self):
        super().__init__()
    
    def _sfmax(self, x: np.ndarray):
        return np.exp(x) / np.sum(np.exp(x))
    
    def _forward(self, x:Tensor):
        y = Tensor(x.shape, x.dtype, x.requires_grad)
        y.data = self._sfmax(x.data)
        return y
    
    def _backward(self, x: Tensor, y: Tensor):
        def bwd(tg):
            tg_shape = tg.shape
            x_shape = x.shape
            tgf = np.reshape(tg, [np.prod(tg_shape)])    # flatten
            xf = np.reshape(x.data, [np.prod(x_shape)])  # flatten
            sxf = self._sfmax(xf)
            grad_sf = np.diag(sxf)
            grad_sf -= np.outer(sxf, sxf)
            tot_grad = np.matmul(tgf, grad_sf)
            return np.reshape(tot_grad, x_shape)
        y.add_parent(x, bwd)

class MatMul(Op):
    def _forward(self, lhs: Tensor, rhs: Tensor) -> Tensor:  # Y = A @ B
        assert(len(lhs.shape) == 2)  # no broadcast
        assert(len(rhs.shape) == 2)
        out_shape = [lhs.shape[0], rhs.shape[1]]
        y = Tensor(out_shape, lhs.dtype, lhs.requires_grad or rhs.requires_grad)
        y.data = np.matmul(lhs.data, rhs.data)
        return y
    
    def _backward(self, lhs: Tensor, rhs: Tensor, y: Tensor):
        y.add_parent(lhs, lambda tg : np.matmul(tg, np.transpose(rhs.data)))  # DA = DY @ DB.T
        y.add_parent(rhs, lambda tg : np.transpose(np.matmul(np.transpose(tg), lhs.data)))  # DB = (DY.T @ DA).T

class SumReduce(Op):
    def __init__(self, axes: Tuple[int]):
        self.axes = axes
    
    def _forward(self, x: Tensor) -> Tensor:
        y_data = np.sum(x.data, self.axes, keepdims=True)
        y = Tensor(y_data.shape, x.dtype, x.requires_grad)
        y.data = y_data
        return y
    
    def _backward(self, x: Tensor, y: Tensor):
        # multiplication with broadcast
        y.add_parent(x, lambda tg : tg * np.ones(x.shape))

class MeanReduce(Op):
    def __init__(self, axes: Tuple[int]):
        self.axes = axes
    
    def _forward(self, x: Tensor) -> Tensor:
        y_data = np.mean(x.data, self.axes, keepdims=True)
        y = Tensor(y_data.shape, x.dtype, x.requires_grad)
        y.data = y_data
        return y
    
    def _backward(self, x: Tensor, y: Tensor):
        n = np.prod(np.array(x.shape)[self.axes])
        alpha = 1.0 / float(n)
        y.add_parent(x, lambda tg : tg * np.ones(x.shape) * alpha)

class MaxReduce(Op):
    def __init__(self, axis: int):
        self.axis = axis
    
    def _forward(self, x: Tensor) -> Tensor:
        y_data = np.max(x.data, self.axis, keepdims=True)
        y = Tensor(y_data.shape, x.dtype, x.requires_grad)
        y.data = y_data
        return y
    
    def _backward(self, x: Tensor, y: Tensor):
        mask = np.copy(x.data)

        def set_one(arr):  # only the maximum value is relevant for the derivative (assumes one maximum)
            max_idx = np.argmax(arr)
            arr[:] = 0
            arr[max_idx] = 1
            return arr

        mask = np.apply_along_axis(set_one, self.axis, mask)

        y.add_parent(x, lambda tg : tg * mask)

class MinReduce(Op):
    def __init__(self, axis: int):
        self.axis = axis
    
    def _forward(self, x: Tensor) -> Tensor:
        y_data = np.min(x.data, self.axis, keepdims=True)
        y = Tensor(y_data.shape, x.dtype, x.requires_grad)
        y.data = y_data
        return y
    
    def _backward(self, x: Tensor, y: Tensor):
        mask = np.copy(x.data)

        def set_one(arr):  # only the minimum value is relevant for the derivative (assumes one minimum)
            min_idx = np.argmin(arr)
            arr[:] = 0
            arr[min_idx] = 1
            return arr

        mask = np.apply_along_axis(set_one, self.axis, mask)

        y.add_parent(x, lambda tg : tg * mask)

class Convolution(Op):
    def __init__(self, ch_in: int, ch_out: int, kernel: tuple, stride: tuple, dilation: tuple = (1, 1)):
        super().__init__()

        self.kernel = kernel
        self.stride = stride
        self.dilation = dilation

        assert(len(self.kernel) == 2)

        # create weight and bias
        self.weight_torch_data = torch.randn((ch_out, ch_in, *self.kernel), requires_grad=True)
        self.bias_torch_data = torch.randn((ch_out), requires_grad=True)
        
        weight_data = self.weight_torch_data.detach().numpy()
        self.weight = Tensor(weight_data.shape, weight_data.dtype, True)
        self.weight.data = weight_data

        bias_data = self.bias_torch_data.detach().numpy()
        self.bias = Tensor(bias_data.shape, bias_data.dtype, True)
        self.bias.data = bias_data

    def _forward(self, x: Tensor) -> Tensor:
        ty = F.conv2d(torch.tensor(x.data), self.weight_torch_data, self.bias_torch_data, self.stride, dilation=self.dilation)
        y_data = ty.detach().numpy()
        y = Tensor(y_data.shape, x.dtype, x.requires_grad)
        y.data = y_data
        return y
    
    def _backward(self, x: Tensor, y: Tensor):
        
        def bwd_x(tg):
            # pad the G tensor internally
            sh, sw = self.stride
            zh, zw = sh - 1, sw - 1
            b, f, dh, dw = tg.shape
            dhp = (dh - 1) * zh + dh
            dwp = (dw - 1) * zw + dw
            g = np.zeros((b, f, dhp, dwp), dtype=np.float32)
            for i in range(dh):
                for j in range(dw):
                    g[:, :, i * sh, j * sw] = tg[:, :, i, j]  # i' = i * s (in explanation)

            # pad G tensor externally with zeros
            _, _, h, w = self.weight.shape
            ah, aw = self.dilation
            wh = (h - 1) * (ah - 1) + h  # pad size
            ww = (w - 1) * (aw - 1) + w
            g = F.pad(torch.tensor(g), (ww-1, ww-1, wh-1, wh-1), "constant")
            
            # rotate the weight
            w_np = self.weight.data[:, :, ::-1, ::-1].copy()
            w = torch.tensor(w_np).transpose(1, 0)
            dy = F.conv2d(g, w, dilation=self.dilation)

            # needs padding with zero if image is smaller (ignored inputs)
            dy = F.pad(dy, (0, x.shape[3] - dy.shape[3], 0, x.shape[2] - dy.shape[2]))
            return dy.detach().numpy()
        
        def bwd_x_alternative(tg):  # transposed conv.
            # calculate output shape
            dy_h = (tg.shape[2] - 1) * self.stride[0] + self.dilation[0] * (self.weight.shape[2] - 1) + 1
            dy_w = (tg.shape[3] - 1) * self.stride[1] + self.dilation[1] * (self.weight.shape[3] - 1) + 1
            
            # calculate output padding
            oh = x.shape[2] - dy_h
            ow = x.shape[3] - dy_w

            # calculate the derivative
            dy = F.conv_transpose2d(
                torch.tensor(tg), 
                self.weight_torch_data, 
                stride=self.stride, 
                dilation=self.dilation,
                output_padding=(oh, ow))
            return dy.detach().numpy()

        def bwd_w(tg):
            w = torch.tensor(tg).transpose(1, 0)
            dy = F.conv2d(torch.tensor(x.data).transpose(1, 0), w, dilation=self.stride, stride=self.dilation)
            dy_np = dy.transpose(1, 0).detach().numpy()  # feature is in w but it has to be the first axis 
            _, _, h, w = self.weight.shape
            return dy_np[:, :, :h, :w]  # can have unused elements on x, here we need to crop the unused elements
        
        def bwd_b(tg):  # tg shape: (batch, feature, y height, y width)
            return tg.sum(axis=(0, 2, 3))

        y.add_parent(x, bwd_x)
        y.add_parent(self.weight, bwd_w)
        y.add_parent(self.bias, bwd_b)
