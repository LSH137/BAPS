import torch
import torchvision

model = torchvision.models.resnet18(pretrained=True)
data= torch.rand(1, 3, 64, 64)
labels = torch.rand(1, 1000)

prediction = model(data) # 순전파 단계
loss = (prediction - labels).sum()
loss.backward() # 역전파 단계

optim = torch.optim.SGD(model.parameters(), lr=1e-2,momentum=0.9) # lr: learning rate
optim.step() # gradient descent
