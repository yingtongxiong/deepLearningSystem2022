"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:

    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):

    def forward(self, x):
        return x

class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype="float32"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.bias = bias
        # initializing parameters
        weight = init.kaiming_uniform(in_features, out_features, nonlinearity="relu", dtype=dtype)
        self.weight = Parameter(weight)
        self.biasflag = bias

        if self.biasflag:
          b = init.kaiming_uniform(out_features, 1, nonlinearity="relu", dtype=dtype).reshape((1, out_features))
          self.bias = Parameter(b)
        
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        out = ops.matmul(X, self.weight)
        if self.bias:
          bias = ops.broadcast_to(self.bias, out.shape)
          out = ops.add(out, bias)
        return out
        ### END YOUR SOLUTION


class Flatten(Module):

    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        n = X.shape[0]
        dim = 1
        for i in range(1, len(X.shape)):
            dim *= X.shape[i]
        return ops.reshape(X, (n, dim))
        ### END YOUR SOLUTION


class ReLU(Module):

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION

class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for m in self.modules:
          x = m(x)
        return x
        ### END YOUR SOLUTION

class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        softmax = ops.LogSumExp(axes=1)(logits)

        batch_size = logits.shape[0]
        num_classes = logits.shape[1]
        one_hot = init.one_hot(num_classes, y)

        z = ops.summation(logits * one_hot, axes=1)
        loss = softmax - z
        total_loss = ops.summation(loss)
        return total_loss / batch_size
        ### END YOUR SOLUTION

class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(self.dim))
        self.bias = Parameter(init.zeros(self.dim))
        self.running_mean = init.zeros(self.dim)
        self.running_var = init.ones(self.dim)
        ### END YOUR SOLUTION


    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        batch_size = x.shape[0]
        broadcast_weight = ops.broadcast_to(ops.reshape(self.weight, (1, -1)), x.shape)
        broadcast_bias = ops.broadcast_to(ops.reshape(self.bias, (1, -1)), x.shape)
        if self.training:
          sums = ops.summation(x, axes=0)
          mean = ops.divide_scalar(sums, batch_size)
          broadcast_mean = ops.broadcast_to(ops.reshape(mean, (1, -1)), x.shape)

          sub = x - broadcast_mean
          sub2 = ops.power_scalar(sub, 2)
          var = ops.summation(ops.divide_scalar(sub2, batch_size), axes=0)
          broadcast_var = ops.broadcast_to(ops.reshape(var, (1, -1)), x.shape)

          # broadcast_weight = ops.broadcast_to(ops.reshape(self.weight, (1, -1)), x.shape)
          # broadcast_bias = ops.broadcast_to(ops.reshape(self.bias, (1, -1)), x.shape)
          
          out = broadcast_weight * (x - broadcast_mean) / ops.power_scalar(broadcast_var + self.eps, 0.5) + broadcast_bias

          self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
          self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        
        else:
          broadcast_running_mean = ops.broadcast_to(ops.reshape(self.running_mean, (1, -1)), x.shape)
          broadcast_running_var = ops.broadcast_to(ops.reshape(self.running_var, (1, -1)), x.shape)
          out = broadcast_weight * (x - broadcast_running_mean) / ops.power_scalar(broadcast_running_var + self.eps, 0.5) + broadcast_bias
        return out
        
        ### END YOUR SOLUTION

class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(self.dim))
        self.bias = Parameter(init.zeros(self.dim))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        batch_size = x.shape[0]
        features = x.shape[1]

        sums = ops.summation(x,axes=1)
        mean = ops.divide_scalar(sums, features)
        tmp = ops.reshape(mean, (-1, 1))
        broadcast_mean = ops.broadcast_to(tmp, x.shape)
        
        sub = x - broadcast_mean
        sub2 = ops.power_scalar(sub, 2)
        var = ops.summation(ops.divide_scalar(sub2, features), axes=1)
        broadcast_var = ops.broadcast_to(ops.reshape(var, (-1, 1)), x.shape)
        
        nominator = ops.power_scalar(broadcast_var + self.eps, 0.5)

        broadcast_weight = ops.broadcast_to(ops.reshape(self.weight, (1, -1)), x.shape)
        broadcast_bias = ops.broadcast_to(ops.reshape(self.bias, (1, -1)), x.shape)
        out = broadcast_weight * (x - broadcast_mean) / nominator + broadcast_bias
        return out 
        ### END YOUR SOLUTION

class Dropout(Module):
    def __init__(self, p = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
          mask = init.randb(*x.shape, p= 1 - self.p) / (1 - self.p)
          return x * mask
        return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(x) + x
        ### END YOUR SOLUTION


