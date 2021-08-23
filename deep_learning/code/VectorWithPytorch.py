import torch

t = torch.FloatTensor([0., 1., 2., 3., 4., 5., 6.])
print(t)

print(f"rank = {t.dim()}") # -> 1
print(f"shape = {t.shape}") # -> print(torch.Size([7]))
print(f"shape = {t.size()}") # -> print(torch.Size([7]))

print("==============================")

print(t[0], t[1], t[-1]) # -> tensor(0.) tensor(1.) tensor(6.)
print(t[2:5], t[4: -1]) # -> tensor([2., 3., 4.]) tensor([4., 5.])