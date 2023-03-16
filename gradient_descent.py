from micrograd.nn import MLP

x = [2.0, 3.0, -1.0]
n = MLP(3, [4,4,1])

print(n(x))

xs = [
  [2.0, 3.0, -1.0],
  [3.0, -1.0, 0.5],
  [0.5, 1.0, 1.0],
  [1.0, 1.0, -1.0]
]
ys = [1.0, -1.0, -1.0, 1.0]

# Gradient Descent

for k in range(20):
  # Forward Pass
  ypred = [n(x) for x in xs]
  loss = sum([(y_pred - y_true)**2 for y_true, y_pred in zip(ys, ypred)])

  # Backward Pass
  for p in n.parameters(): #setting grads to 0
    p.grad = 0.0
  loss.backward()

  # Update
  for p in n.parameters():
    p.data += -0.05 * p.grad

  print(k, loss.data)

print(ypred)