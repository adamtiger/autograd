"""
    Here several tests, examples can be found for 
    checking the back-propagation.
"""

import numpy as np
import tensor
import graph
import loss
import ops

from jax import grad
import jax.numpy as jnp
import jax.scipy.special as jsc

from torch.nn import functional as F
import torch


def convolution_grad_example():
    
    # parameters
    xb = 2
    xc = 5
    xh = 15
    xw = 15

    kernel = (2, 3)
    stride = (2, 2)
    dilation = (2, 3)

    yc = 3

    # build graph for convolution
    class ExampleGraph(graph.Graph):
        def __init__(self):
            super().__init__()
            self.conv = ops.Convolution(xc, yc, kernel, stride, dilation)

        def forward(self, x):
            y = self.conv(x)
            return y

    # backward for autograd
    sg = ExampleGraph()
    x = tensor.rand([xb, xc, xh, xw], np.float32)
    y = sg.forward(x)
    
    yh = int((xh - (kernel[0] - 1) * dilation[0] - 1) / stride[0]) + 1
    yw = int((xw - (kernel[1] - 1) * dilation[1] - 1) / stride[1]) + 1
    y_real = tensor.rand([xb, yc, yh, yw], np.float32)

    mse_loss = loss.MSE()
    l = mse_loss(y_real, y)
    l.backward()

    # torch implementation
    def torch_graph(x):
        w = sg.conv.weight_torch_data
        b = sg.conv.bias_torch_data
        stride = sg.conv.stride
        dilation = sg.conv.dilation
        y = F.conv2d(x, w, b, stride, dilation=dilation)

        y_diff = y - torch.tensor(y_real.data)
        torch_mse_loss = torch.mean(torch.square(y_diff))
        return torch_mse_loss
    
    x_torch = torch.tensor(x.data, requires_grad=True)
    tloss = torch_graph(x_torch)
    tloss.backward()

    # compare the backward tensors
    def calc_total_diff(dy_autograd, dy_expected):
        return np.sqrt(np.square(dy_autograd - dy_expected).sum())

    dw = calc_total_diff(sg.conv.weight.total_grad, sg.conv.weight_torch_data.grad.detach().numpy())
    dx = calc_total_diff(x.total_grad, x_torch.grad.detach().numpy())
    db = calc_total_diff(sg.conv.bias.total_grad, sg.conv.bias_torch_data.grad.detach().numpy())

    print("Convolution - difference in grads: ")
    print(f"  DW: {dw}")
    print(f"  DX: {dx}")
    print(f"  DB: {db}")


def gelu_grad_example():
    
    # build graph for convolution
    class ExampleGraph(graph.Graph):
        def __init__(self):
            super().__init__()
            self.sr = ops.SumReduce(axes=(1))
            self.gelu = ops.GeLU()
 
        def forward(self, x):
            y0 = self.sr(x)
            y = self.gelu(y0)
            return y

    # backward for autograd
    sg = ExampleGraph()
    x = tensor.rand([2, 3, 3], np.float32)
    y = sg.forward(x)

    y_real = tensor.rand([2, 1, 3], np.float32)

    mse_loss = loss.MSE()
    l = mse_loss(y_real, y)
    l.backward()

    # jax implementation
    def jax_graph(x):
        y0 = jnp.sum(x, (1), keepdims=True)
        y = 0.5 * y0 * (1.0 + jsc.erf(y0 / jnp.sqrt(2.0)))  # gelu implementation
        y_diff = jnp.subtract(y_real.data, y)
        jax_mse_loss = jnp.mean(jnp.square(y_diff))
        return jax_mse_loss
    
    grad_jax_sgraph = grad(jax_graph)
    x_jax_grad = grad_jax_sgraph(x.data)

    # compare the backward tensors
    def calc_total_diff(dy_autograd, dy_expected):
        return np.sqrt(np.square(dy_autograd - dy_expected).sum())

    dx = calc_total_diff(x.total_grad, x_jax_grad)

    print("Gelu - difference in grads: ")
    print(f"  DW: {dx}")
    

def matmul_grad_example():
    
    # build graph for convolution
    class ExampleGraph(graph.Graph):
        def __init__(self):
            self.sin = ops.Sinus()
            self.mm = ops.MatMul()
            self.relu = ops.ReLU()
            self.exp = ops.Exp()
            self.reshape = ops.Reshape([2, 3])
 
        def forward(self, x):
            y0 = self.exp(x)
            y1 = self.sin(x)
            y2 = self.reshape(self.relu(y0))
            y = self.mm(y1, y2)
            return y

    # backward for autograd
    sg = ExampleGraph()
    x = tensor.rand([3, 2], np.float32)
    y = sg.forward(x)

    y_real = tensor.rand([3, 3], np.float32)

    mse_loss = loss.MSE()
    l = mse_loss(y_real, y)
    l.backward()

    # jax implementation
    def jax_graph(x):
        y0 = jnp.exp(x)
        y1 = jnp.sin(x)
        y2 = jnp.reshape(jnp.maximum(y0, 0), [2, 3])
        y = jnp.matmul(y1, y2)
        y_diff = jnp.subtract(y_real.data, y)
        jax_mse_loss = jnp.mean(jnp.square(y_diff))
        return jax_mse_loss
    
    grad_jax_sgraph = grad(jax_graph)
    x_jax_grad = grad_jax_sgraph(x.data)

    # compare the backward tensors
    def calc_total_diff(dy_autograd, dy_expected):
        return np.sqrt(np.square(dy_autograd - dy_expected).sum())

    dx = calc_total_diff(x.total_grad, x_jax_grad)

    print("Matmul - difference in grads: ")
    print(f"  DW: {dx}")
