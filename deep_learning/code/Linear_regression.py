import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

cost = float()
epoch = 0

#torch.manual_seed(1)

x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])

x2_train = torch.FloatTensor([[1], [2], [3]])
y2_train = torch.FloatTensor([[3], [6], [9]])

# 가중치 W를 0으로 초기화하고 학습을 통해 값이 변경되는 변수임을 명시함
W = torch.zeros(1, requires_grad=True)
# requires_grand=True -> 학습을 통해 값이 변경되는 변수임을 의미함

b = torch.zeros(1, requires_grad=True)

for _ in range(2000):
    epoch += 1
    
	# 가설 세우기
    hypothesis = x_train * W + b

	# define loss function
    cost = torch.mean((hypothesis - y_train)**2)
    # print(f"loss = {cost}")

	# 경사하강법 구현하기
    optimizer = optim.SGD([W, b], lr=0.01)
	# lr: 학습률

	# gradient를 0으로 초기화
    # Pytorch는 미분으로 얻은 기울기를 이전에 계산된 기울기에 누적시키기 때문
    optimizer.zero_grad()
    
	# 비용함수를 미분하여 gradient계산
    cost.backward()
    
	# W와 b를 업데이트
    optimizer.step()
    
    if(epoch % 100 == 0):
        print(f"epoch: {epoch}/2000 | W: {W.item()} | b: {b.item()} | error: {cost}")


epoch = 0  
for _ in range(2000):
    epoch += 1
    
	# 가설 세우기
    hypothesis = x2_train * W + b

	# define loss function
    cost = torch.mean((hypothesis - y2_train)**2)
    # print(f"loss = {cost}")

	# 경사하강법 구현하기
    optimizer = optim.SGD([W, b], lr=0.01)
	# lr: 학습률

	# gradient를 0으로 초기화
    # Pytorch는 미분으로 얻은 기울기를 이전에 계산된 기울기에 누적시키기 때문
    optimizer.zero_grad()
    
	# 비용함수를 미분하여 gradient계산
    cost.backward()
    
	# W와 b를 업데이트
    optimizer.step()
    
    if(epoch % 100 == 0):
        print(f"epoch: {epoch}/2000 | W: {W.item()} | b: {b.item()} | error: {cost}")


# 두 번째 데이터셋으로 모델이 바뀐다. 첫 번째 데이터셋에 대한 기록이 남지 않는다.