import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

x_train = torch.FloatTensor([[73, 80, 75],
                            [93, 88, 93],
                            [89, 91, 90],
                            [96, 98, 100],
                            [73, 66, 70]])

y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

dataset = TensorDataset(x_train, y_train)
# dataset으로 저장

dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
# 통상적으로 2개의 인자를 받는다(데이터셋, 배치크기).
# suffle=True로 Epoch마다 데이터셋을 섞어서 데이터가 학습되는 순서가 바뀐다

model = nn.Linear(3, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-8)

nb_epochs = 100000
for epoch in range(nb_epochs+1):
    for batch_idx, samples, in enumerate(dataloader):
        # print(batch_idx)
        # print(samples)
        
        x_train, y_train = samples
        prediction = model(x_train)
        
        cost = F.mse_loss(prediction, y_train)
        
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        
        print(f'Epoch: {epoch}, Batch: {nb_epochs}, cost: {cost}')
        
        
new_var = torch.FloatTensor([[73, 80, 75]])
pred_y = model(new_var)
print("훈련 후 입력이 73, 80, 75일때 예측값: ", pred_y)  