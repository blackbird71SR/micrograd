import torch
from micrograd.engine import Value

def test_sanity_check():
  x = Value(-4.0)
  z = 2 * x + 2 + x
  q = z.relu() + z * x
  h = (z * z).relu()
  y = h + q + q * x
  
  y.backward()
  xmg, ymg = x, y

  x = torch.Tensor([-4.0]).double()
  x.requires_grad = True
  z = 2 * x + 2 + x
  q = z.relu() + z * x
  h = (z * z).relu()
  y = h + q + q * x
  
  y.backward()
  xpt, ypt = x, y

  # Forward pass is correct
  assert ymg.data == ypt.data.item()
  # Backward pass is correct
  assert xmg.grad == xpt.grad.item()

test_sanity_check()