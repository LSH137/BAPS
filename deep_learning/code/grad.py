import torch

w = torch.tensor(2.0, requires_grad=True)

y = w**2
z = 2*y + 5

z.backward()
print(f"dz/dw = {w.grad}")
print(f"dz/dy = {y.retain_grad()}") # 가장 기본이 되는 변수에 대해서만 미분 가능
