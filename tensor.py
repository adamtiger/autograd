import numpy as np
from typing import List

class Tensor:
    """
        A computational graph has two main parts:
        - operators (nodes of the graph)
        - tensors (edges of the graph)

        The tensor contains the data and related information
        like shape and data type.
        The data here is a numpy ndarray (for simplicity).

        For gradient calculations there are two typical options:
        - build a separate graph for backpropagation (with special backward operators)
        - record the gradients inside the tensors itself

        The last approach is more flexible and avoids building
        a separate graph (simplicity). The backward propagation can 
        be done automatically with the same mechanism applied in each
        tensor.

        But we have to clearly identify the source tensor(s) where
        the back propagation starts. (In a dnn model, most likely a
        loss function will produce the last tensor.)
    """
    def __init__(self, shape: list, dtype: np.dtype, requires_grad: bool = True):
        self.shape = shape
        self.dtype = dtype
        self.requires_grad = requires_grad
        # buffer
        self.data = np.zeros(shape, dtype)
        # edges
        self.parents: List[Tensor] = list()
        self.num_children = 0
        # gradients
        self.grad_funcs: List[any] = list()  # parent gradient calculation
        self.total_grad: np.ndarray = np.zeros(shape, dtype)  # resulting gradient after derivation
        self.num_visits = 0  # should be equal with num children, when backprop wants to start
    
    def __str__(self):
        return str(self.data)
    
    def set_no_grad(self):
        self.requires_grad = False

        self.parents.clear()
        self.grad_funcs.clear()
        self.total_grad = None

    def add_parent(self, tensor, grad: any):
        if tensor.requires_grad:
            self.parents.append(tensor)
            self.grad_funcs.append(grad)
            tensor.num_children += 1
    
    def backward(self):  # propagate backward (chain rule) and calculates all the gradients in the tensors
        # backward bfs for backprop
        self.num_visits += 1  # to calculate the total gradient, the gradients from all the children is needed
        if self.num_children <= self.num_visits:  
            for p, gfn in zip(self.parents, self.grad_funcs):
                p.total_grad += gfn(self.total_grad)
            for p in self.parents:
                p.backward()     # continue backward prop on parents
            self.num_visits = 0  # reset, for next backprop


def rand(shape: list, dtype: np.dtype):  # helper func., generating tensors easily
    t = Tensor(shape, dtype)
    t.data = np.random.rand(*shape).astype(dtype)
    return t
