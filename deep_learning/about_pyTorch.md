# PyTorch

***

official reference: https://tutorials.pytorch.kr/beginner/blitz/tensor_tutorial.html   

reference: https://wikidocs.net/book/2788   

## Pytorch 패키지의 기본 구성

### torch

> 메인 네임스페이스. 텐서 등의 다양한 수학 함수가 포함됨. Numpy와 유사한 구조를 가짐   



### torch.autograd

> 자동 미분을 위한 함수들이 포함되어 있음. 자동 미분의 on/off를 제어하는 콘텍스트 매니저(enable_grand/no_grand)나 자체 미분 가능 함수를 정의할 때 사용하는 기반 클래스인 'Function' 등이 포함됨   



### torch.nn

> 신경망을 구축하기 위한 다양한 데이터 구조나 레이더 등이 정의되어있다. RNN, LSTM과 같은 레이어, ReLU와 같은 활성화 함수, MSELoss와 같은 손실함수들이 있다.   



### torch.optim

> 확률적 경사 하강법(Stochastic Gradient Descant, SGD)를 중심으로 한 파라미터 최적화 알고리즘이 구현되어 있음   



### torch.utils.data

> SGD의 반복 연산을 실행할 때 사용하는 미니 배치용 유틸리티 함수가 포함되어져 있다.



### torch.onnx

> ONNX(Open Neural Network Exchange)의 포맷으로 모델을 export할 때 사용한다. ONNX는 서로 다른 딥러닝 프레임워크 간에 모델을 공유할 때 사용하는 포맷이다.   



## 텐서 조작하기

### Vector, Matrix, Tensor

#### vector, Matrix, Tensor

> vector = 1차원 Tensor   
>
> matrix = 2차원 Tensor   



#### 텐서의 모양 (PyTorch Tensor Shape Convention)    

> - 2D Tensor  
>
>   (텐서의 크기) = batch size * dimension   
>
>   batch size: 훈련 데이터 개수(행 수)   
>
>   dimension: 훈련 데이터 하나가 가지고 있는 정보 수(열 수 )   
>
> - 3D Tensor   
>
>   (텐서의 크기) = batch size * dim * width   

#### numpy로 텐서 만들기

> * 1D
>
>   [숫자, 숫자, 숫자] 의 형식으로 만들고 이것을 np.array()로 감싸주면 된다   
>
>   Numpy의 인덱스는 지금 알고있는 파이썬 인덱스와 동일하다   
>
>   슬라이싱도 가능하다   
>
>   t = [0, 1, 2, 3, 4, 5, 6]
>
>   -> rank = 1
>
>   -> shape =(7, )
>
> * 2D
>
>   t = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])     
>
>   -> rank = 2	// t.ndim 으로 알아낸다      
>
>   -> shape = (4, 3)	//t.shape 으로 알아낸다     

#### PyTorch Tensor Allocation   

>t.dim()을 이용하여 차원을 알아낸다    
>
>t.size()를 이용하여 크기를 확인한다      
>
>* 1D   
>
> numpy, python과 동일   
>
>* 2D   
>
>t = torch.FloatTensor([1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [10., 11.,12.])      
>
>t[0:2] -> [[1., 2., 3.], [4., 5., 6.]]   
>
>t[0:2, 1] -> t[0:2]에 포함된 리스트 중 인덱스가 1인 것들만 뽑아온다   

#### Broadcasting

>행렬의 연산에서 행렬의 크기에 대한 제한이 있다. 딥러닝할때 불가피하게 이 수학적 규칙을 벗어날 수 있으므로, 파이토치에서는 자동으로 크기를 맞추어 연산을 수행하는 브로드캐스팅이라는 기능을 제공한다.   
>
>* 같은 크기의 행렬을 연산   
>```python
>  m1 = torch.FloatTensor([[3, 3]])   
>  m2 = torch.FloatTensor([[2, 2]])   
>  print(m1 + m2) # tensor([[5., 5.]])   
>```
>* 다른 크기의 행렬을 이용한 연산   
>```python
>  m1 = torch.FloatTensor([[1, 2]])   
>  m2 = torch.FloatTensor([3]) # [3, 3]으로 연산된다   
>  print(m1 + m2) # tensor([[4., 5.]])   
>```
> -----------------------------------------------------------
>```python
>  m1 = torch.FloatTensor([[1, 2]])   
>  m2 = torch.FloatTensor([[3], [4]])   
>  print(m1 + m2)# [1, 2], [1, 2] + [3, 3], [4, 4] 로 연산 -> tensor([4., 5.], [5., 6])   
> ```

#### Matrix Multiplication Vs. Multiplication

>행렬 곱셈: .matmul    
>
>```python
>m1 = torch.FLoatTensor([[1, 2], [3, 4]])
>m2 = torch.FloatTensor([[1], [2]])
>
>print("Shape of Matrix 1:  ", m1.shape)
>print("Shape of Matrix 2: ", m2.shape)
>print(m1.matmul(m2))
>#tensor([[ 5.],
>#        [11.]])
>```
>
>원소 별 곱셈: .mul  
>
>```python
>m1 = torch.FloatTensor([[1, 2], [3, 4]])
>m2 = torch.FloatTensor([[1], [2]])
>print('Shape of Matrix 1: ', m1.shape) # 2 x 2
>print('Shape of Matrix 2: ', m2.shape) # 2 x 1
>print(m1 * m2) # 2 x 2
>print(m1.mul(m2))
>
>#tensor([[1., 2.],
>#        [6., 8.]])
>#tensor([[1., 2.],
>#        [6., 8.]])
>```
>
>-> Broadcasting이 적용됨   

#### mean   

>```python
>t = torch.FloatTensor([[1, 2], [3, 4]])
>print(t.mean()) -> tensor(2.5000)
>t.mean(dim=0) # 입력에서 첫 번째 차원을 제거한다 -> 결과가 행이 제거되고 열만 있는 벡터
>
>#1과 3의 평균을 구하고, 2와 4의 평균을 구한다 -> [2., 3.]
>t.mean(dim=1) # 입력에서 두 번째 차원을 제거한다 -> 결과가 열이 제거되고 행만 있는 벡터
>t.mean(dim=-1) # 마지막 차원을 결과에서 제거
>```

#### sum

>```python
>t = torch.FloatTensor([[1, 2], 
>                       [3, 4]])
>print(t)
>
>print(t.sum()) # 단순히 원소 전체의 덧셈을 수행 -> 10
>print(t.sum(dim=0)) # 행을 1개로 만든 결과를 반환 -> [4, 6]
>print(t.sum(dim=1)) # 열을 1개로 만든 결과를 반환 -> [3, 7]
>print(t.sum(dim=-1)) # 마지막 차원을 1개로 만든 결과를 반환 -> [3, 7]
>```

#### 최대(Max)와 ArgMax

>Max는 원소의 최댓값을 반환하고, ArgMax는 최대값을 가진 인덱스를 리턴한다.
>
>```python
>t = torch.FloatTensor([[1, 2], 
>                       [3, 4]])
>print(t)
>print(t.max()) #->tensor(4.)
>print(t.max(dim=0)) #첫 번째 차원을 1줄로(행을 1줄로 -> 열끼리 비교)
>#-> tensor([3., 4.]), tensor([1, 1])
>#max에 dim 인자를 주면 argmax도 같이 반환한다.
>#argmax -> 3이 첫 번째 열에서 1의 인덱스를 가진다, 4가 두번째 열에서 1의 인덱스를 가진다
>print('Max: ', t.max(dim=0)[0]) # max값만 가져온다
>print('Argmax: ', t.max(dim=0)[1]) # argmax값만 가져온다
>```

#### View

>원소의 수를 유지하면서 텐서의 크기를 변경. 중요
>
>Numpy의 Reshape와 같은 역할
>
>```python
>t = np.array([[[0, 1, 2], 
>               [3, 4, 5]], 
>              [[6, 7, 8], 
>               [9, 10, 11]]]) # 3차원 텐서
>ft = torch.FloatTensor(t)
>print(ft.shape) # torch.Size([2, 2, 3])
>```
>
>* 3차원 텐서를 2차원 텐서로 변경
>
>  ```python
>  print(ft.view([-1, 3])) # ft라는 텐서를 (?, 3)의 크기로 변경
>  # -> tensor([[0., 1., 2.], 
>  #			[3., 4., 5.], 
>  #			[6., 7., 8.], 
>  #			[9., 10., 11.]])
>  print(ft.view([-1, 3].shape)) # torch.Size([4, 3])
>  ```
>
>  view([-1, 3]): -1은 첫 번째 차원은 파이토치가 알아서 하는걸로. 3은 두번째 차원의 길이를 3이 되도록 하라는 의미.
>
>  * view는 기본적으로 변경 전과 후의 원소 개수가 유지되어야 함
>  * 파이토치의 view는 사이즈가 -1로 설정되면 다른 차원으로 해당 값을 유추
>
>* 3차원 텐서에서 3차원 텐서로 크기 변경
>
>  ```python
>  print(ft.view([-1, 1, 3]))
>  # tensor([[[0., 1., 2.]], 
>  #		[[3., 4., 5.]], 
>  #		[[6., 7., 8.]], 
>  #		[[9., 10., 11.]]])
>  print(ft.view([-1, 1, 3]).shape) # torch.Size([4, 1, 3])
>  ```
>
>  

#### Squeeze

>차원이 1인 경우에는 해당 차원을 제거한다.
>
>```python
>ft = torch.FloatTensor([[0], [1], [2]])
>print(ft)
>print(ft.shape)
>#tensor([[0.],
>#        [1.],
>#        [2.]])
>#torch.Size([3, 1])
>print(ft.squeeze())
>print(ft.squeeze().shape)
>#tensor([0., 1., 2.])
>#torch.Size([3])
>```

#### Unsqueeze

>특정 위치에 1인 차원을 추가한다 (첫 번째 차원 -> 인덱스 0)
>
>```python
>ft = torch.Tensor([0, 1, 2])
>print(ft.shape) # torch.Size([3])
>print(ft.unsqueeze(0)) # 인덱스가 0번부터 시작하므로 0은 첫번째 차원을 의미
># tensor([[0., 1., 2.]])
>print(ft.unsqueeze(0).shape) # torch.Size([1,3])
>
>print(ft.unsqueeze(-1)) # -1은 마지막 인덱스 -> 맨 뒤에 차원 추가
># tenosr([[0.], 
>#		[1.], 
>#		[2.]])
>print(ft.unsqueeze(-2).shape) # torch.Size([3, 1])
>```
>
>view로도 구현 가능하다
>
>2차원으로 바꾸고 첫 번째 차원은 1로 바꾼다
>
>```python
>print(ft.view(1, -1)) # tensor([[0., 1., 2.]])
>print(ft.view(1, -1).shape) # torch.Size([1, 3])
>```

#### Type Casting

> * 텐서에는 자료형이 있다
> * GPU연산을 위한 자료형도 있다 (torch.cuda.FloatTensor)
>
> ```python
> lt = torch.LongTensor([1, 2, 3, 4])
> print(lt)
> print(lt.float()) # float형으로 type casting->tensor([1., 2., 3., 4.])
> 
> bt = torch.ByteTensor([True, False, False, True])
> print(bt) # tensor([1, 0, 0, 1], dtype=torch.uint8)
> print(bt.long()) # tensor([1, 0, 0, 1])
> print(bt.float()) # tensor([1., 0., 0., 1.])
> ```

#### Concatenate

> 두 텐서를 연결하는 방법 - torch.cat([ ]) 사용
>
> ```python
> x = torch.LongTensor([[1, 2], [3, 4]])
> y = torch.Longtensor([[5, 6], [7, 8]])
> 
> print(torch.cat([x, y], dim=0)) # 첫 번째 차원을 늘려라
> # tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
> ```
>
> * 원래 텐서를 변경하지는 않는다. 합쳐진 텐서를 반환할 뿐
> * 자료형이 다른 두 텐서를 연결할경우 더 유효자리가 많은 텐서의 자료형을 따라간다.

#### Stacking

> 연결하는 다른 방법
>
> ```python
> x = torch.LongTensor([1, 4])
> y = torch.LongTensor([2, 5])
> z = torch.LongTensor([3, 6])
> 
> print(torch.stack([x, y, z]))
> # tensor([[1, 4], [2, 5], [3, 6]])
> 
> print(torch.stack([x, y, z], dim=1)) # 두 번째 차원이 증가하도록 쌓아라
> # tensor([[1, 2, 3], [4, 5, 6]])
> ```

#### ones_like, zeros_like

> * ones_like: 1로 채워진 텐서를 만든다
> * zeros_like: 0으로 채워진 텐서를 만든다
>
> ```python
> x = torch.FloatTensor([[0, 1, 2], [2, 1, 0]])
> print(x)
> # tensor([0., 1., 2.], [2., 1., 0.])
> 
> print(torch.ones_like(x)) # 입력 텐서와 크기를 통일하게 하면서 값을 1로 채운다
> # tensor([[1., 1., 1.], [1., 1., 1.]])
> 
> print(torch.zeros_like(x)) # 입력 텐서와 크기를 동일하게 하면서 값을 0으로 채운다
> # tensor([0., 0., 0.,], [0., 0., 0.])
> ```

#### In-place Operation

> 덮어쓰기 연산
>
> ```python
> x = torch.FLoatTensor([[1, 2], [3, 4]])
> print(x.mul(2.)) # 곱하기 2를 한 결과
> # -> tensor([[2., 4.], [6., 8.]])
> 
> print(x) # 원래 텐서 출력
> # -> tensor([[1., 2.], [3., 4.]])
> ```
>
> 값이 덮어씌워지지 않았다
>
> 이때 mul_()를 이용하면 값이 덮어씌워진다.
>
> ```python
> print(x.mul_(2.)) # 2를 곱한 결과를 x에 저장후 출력
> # -> tensor([[2., 4.], [6., 8.]])
> 
> print(x)
> # -> tensor([[2., 4.], [6., 8.]])
> ```



## Function and Class

파이썬의 function과 class와 동일함



## Linear Regression

### Linear Regression

#### 데이터에 대한 이해 (Data Definition)

>* 훈련 데이터셋(training dataset)의 구성   
>
>  예측을 위해 사용하는 데이터   
>
>  모델을 학습시키기 위한 데이터는 pytorch 텐서의 형태를 가져야 한다.   
>
>  입력과 출력을 각기 다른 텐서에 저장할 필요가 있다. 주로 x와 y사용   
>
>  ```python
>  x_train = torch.FloatTensor([[1], [2], [3]])
>  y_train = torch.FloatTensor([[2], [4], [6]])
>  ```
>
>* 가설 수립
>
>  머신러닝에서 식을 세울때 이 식을 가설(Hypothesis)라고 한다.
>
>  맞는 가설이 아니라고 판단되면 계속 수정해나가게 되는 식
>
>  선형회귀의 경우 선형이므로 H(x) = Wx + b와 같은 식 사용
>
>  * W: 가중치 (weight)
>  * b: 편향 (bias)
>
>* Cost function에 대한 이해
>
>  = loss function(손실 함수) = error function = objective function(목적 함수)
>
>  오차는 MSE(Mean Squared Error)로 구한다. sigma[ ( y_i - H(x_i) )^2 ] / n
>
>* 옵티마이저 - Gradient Descent
>
>  W와 b에 따른 cost는 빗살무늬 토기 모양의 그래프를 그린다 -> 경사하강법으로 토기의 맨 밑바닥을 찾는 것
>
>  * 임의의 초기값 W와 b를 정한다
>  * 오차의 밑바닥을 향해서 조금씩 다가간다
>  * 오차가 최소화 되는 지점은 접선의 기울기가 0이 되는 지점이며 미분값이 0이 되는 지점이다.

#### PyTorch로 구현하기

> ```python
> import torch
> import torch.nn as nn
> import torch.nn.functional as F
> import torch.optim as optim
> 
> float cost = 0.0
> int epoch = 0
> 
> torch.manual_seed(1)
> 
> x_train = torch.FloatTensor([[1], [2], [3]])
> y_train = torch.FloatTensor([[2], [3], [4]])
> 
> # 가중치 W를 0으로 초기화하고 학습을 통해 값이 변경되는 변수임을 명시함
> W = torch.zeros(1, requires_grad=True)
> # requires_grand=True -> 학습을 통해 값이 변경되는 변수임을 의미함
> 
> b = torch.zeros(1, requires_grad=True)
> 
> while(cost < 0.001):
> 	epoch += 1
>     
> 	# 가설 세우기
> 	hypothesis = x_train * W + b
> 
> 	# define loss function
> 	cost = torch.mean((hypothesis - y_train)**2)
> 	print(f"loss = {cost}")
> 
> 	# 경사하강법 구현하기
> 	optimizer = optim.SGD([W, b], lr=0.01)
> 	# lr: 학습률
> 
> 	# gradient를 0으로 초기화
>     # Pytorch는 미분으로 얻은 기울기를 이전에 계산된 기울기에 누적시키기 때문
> 	optimizer.zero_grad()
>     
> 	# 비용함수를 미분하여 gradient계산
> 	cost.backward()
>     
> 	# W와 b를 업데이트
> 	optimizer.step()
>     
>     if(epoch % 10 == 0):
>     	print(f"epoch: {epoch} | W: {W.item()} | b: {b.item()}")
> 	
> ```

#### 자동 미분(Autograd) 실습

> ```python
> import torch
> 
> w = torch.tensor(2.0, requires_grad=True)
> 
> y = w**2
> z = 2*y + 5
> 
> # w에 대한 기울기를 계산한다
> z.backward()
> 
> # 해당 수식을 w에 대해 미분한 값을 출력
> print("dz/dw = {w.grad}")
> # dz/dy는 계산할 수 없다
> ```

#### Multivariable Linear regression

> 여러 개의 x값으로부터 y값을 예측한다.
>
> 예: 3개의 퀴즈 점수로부터 최종 점수를 예측하는 모델 제작
>
> H(x) = w1x1 + w2x2 + w3x3 + b1 + b2 + b3-> 두 행렬의 내적으로 표현 가능
>
> -> H(x) = XW + B
>
> ```python
> import torch
> import torch.nn as nn
> import torch.nn.functional as F
> import torch.optim as optim
> 
> x_train = torch.FloatTensor([[73, 80,75], 
>                              [93, 88, 93], 
>                              [89, 91, 80], 
>                              [96, 98, 100], 
>                              [73, 66, 70]])
> 
> y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])
> 
> W = torch.zeros((3, 1), requires_grad=True)
> b = torch.zeros(1, requires_grad=True)
> 
> optimizer = optim.SGD([W, b], lr=1e-5)
> 
> # model = nn.Linear(3, 1) # input_dim = 3, output_dim=1
> 
> for epoch in range(1000):
> 	hypothesis = x_train.matmul(W) + b
>     # prediction = model(x_train) 과 동일
>     
>     cost = torch.mean((hypothesis - y_train)**2)
>     # cost = F.mse_loss(prediction, y_train)과 동일
>     
>     optimizer.zero_grad()
>     cost.backward()
>     optimizer.step()
>     
>     print(f"epoch: {epoch} | hypothesis: {hypothesis.squeeze().detach()} | cost: {cost.item()}")
>     
> 
> new_var = torch.FLoatTensor([[73, 80, 75]])
> pred_y = model(new_var)
> print(f"예측값: {pred_y}")
> 
> #모델에 쓰인 W와 b값 출력
> print(list(model.parameters()))
> 
> ```

#### class 를 이용한 다중 선형 회귀 구현

> ```python
> import torch
> import torch.nn as nn
> import torch.nn.functional as F
> 
> class MultivariateLinearRegressionModel(nn.Module):
>     def __init__(self):
>         super().__init()
>         self.linear = nnLinear(3, 1)
>        
>     def forward(self, x):
>         return self.linear(x)
> 
> if __name__ == __main__:
>     #data
> 	x_train = torch.FloatTensor([[73, 80,75], 
>     	                         [93, 88, 93], 
>         	                     [89, 91, 80], 
>             	                 [96, 98, 100], 
>                 	             [73, 66, 70]])
> 
> 	y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])
>     
>     #model
>     model = MultivariateLinearRegressionModel()
>     optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)
>     
>     for epoch in range(2000):
>         prediction = model(x_train)
>         
>         cost = F.mse_loss(prediction, y_train)
>         optimizer.zero_grad()
>         cost.backward()
>         optimizer.step()
>         
>         if epoch % 100 == 0:
>             print(f"Epoch: {epoch}/2000 | cost: {cost.item()}")
> 
> ```
>
> 
