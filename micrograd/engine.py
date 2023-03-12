import math

class Value:

  def __init__(self, data, _children=(), _op=''):
    self.data = data
    self.grad = 0
    # internal variables for autograd graph construction
    self._backward = lambda: None
    self._prev = set(_children)
    self._op = _op

  def __add__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data + other.data, (self, other), '+')

    def _backward():
      self.grad += out.grad
      other.grad += out.grad 
    out._backward = _backward

    return out

  def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data * other.data, (self, other), '*')

    def _backward():
      self.grad += other.data * out.grad
      other.grad += self.data * out.grad
    out._backward = _backward

    return out

  def __pow__(self, other):
    assert isinstance(other, (int, float), "only supporting int/float powers for now")
    out = Value(self.data ** other, (self,), f'**{other}')

    def _backward():
      self.grad += (other * (self.data ** (other-1))) ** out.grad
    out._backward = _backward

    return out

  def relu(self):
    out = Value(0 if self.data < 0 else self.data, (self,), f'ReLU')

    def _backward():
      self.grad += (out.data > 0) * out.grad
    out._backward = _backward

    return out

  def tanh(self):
    x = self.data
    t = (math.exp(2*x)-1)/(math.exp(2*x)+1)
    out = Value(t, (self,), 'tanh')

    def _backward():
      self.grad += (1-t**2) * out.grad
    out._backward = _backward

    return out