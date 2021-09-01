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
> class MultivariateLinearRegressionModel(nn.Module): # nn.Module를 상속
>  def __init__(self):
>      super().__init() # nn.Module클래스의 속성을 가지고 초기화
>      self.linear = nnLinear(3, 1)
> 
>  def forward(self, x): # 모델이 학습 데이터를 입력받아서 forward연산을 진행
>      return self.linear(x)
>     
>  # forward()함수는 model이 객체를 데이터와 함께 호출하면 자동으로 실행된다.
> 
> if __name__ == __main__:
>  #data
> 	x_train = torch.FloatTensor([[73, 80,75], 
>  	                         [93, 88, 93], 
>      	                     [89, 91, 80], 
>          	                 [96, 98, 100], 
>              	             [73, 66, 70]])
> 
> 	y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])
> 
>  #model
>  model = MultivariateLinearRegressionModel()
>  optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)
> 
>  for epoch in range(2000):
>      prediction = model(x_train)
> 
>      cost = F.mse_loss(prediction, y_train)
>      optimizer.zero_grad()
>      cost.backward()
>      optimizer.step()
> 
>      if epoch % 100 == 0:
>          print(f"Epoch: {epoch}/2000 | cost: {cost.item()}")
> 
> ```
>



### Mini Batch and Data Load

#### Mini Batch and Batch Size

> * 데이터 샘플의 개수가 많아질 때 전체 데이터를 작은 단위로 나누어서 해당 단위로 학습하는 개념
> * 이 단위를 Mini Batch
> * Mini Batch만큼 가져와서 cost를 계산하고 경사하강을 수행함
> * 다름 Mini Batch를 가져와서 cost를 계산하고 경사하강을 수행함
> * 이것을 마지막 Mini Batch까지 반복
> * 이렇게 전체 데이터에 대한 학습이 1회 끝나면 1Epoch가 끝난다
> * Mini Batch의 개수는 결국 Mini Batch의 크기를 얼마로 하느냐에 따라 달라진다
> * Mini Batch의 크기를 batch size라고 한다
> * 전체 데이터에 대해 한번에 경사하강 수행 -> 배치 경사 하강법
> * Mini Batch 단위로 경사하강 -> Mini Batch 경사 하강법
> * 배치 경사 하강법은 최적값에 수렴하는 과정이 매우 안정적
> * Mini Batch 경사 하강법은 최적값으로 수렴하는 과정에서 헤메기도 함. 훈련 속도 빠름
> * Batch Size는 보통 2의 제곱수 사용. CPU, GPU 메모리가 2의 배수이기 때문

#### Iteration

> 이터레이션은 한 번의 epoch안에서 이루어지는 매기변수인 가중치 W와b의 업데이트 횟수.  전체 데이터가 2000일때 배치 크기를 200으로 한다면 이터레이션의 수는 10

#### Data Load

> pytorch는 Dataset과 DataLoader를 제공한다.  이것으로 미니 배치 학습, 데이터 셔플, 병렬처리까지 간단히 수행할 수 있다.
>
> 기본적으로 Dataset를 정의하고 이것을 DataLoader에 전달하는 것으로 구현
>
> ```python
> import torch
> import torch.nn as nn
> import torch.nn.functional as F
> 
> from torch.utils.data import TensorDataset
> from torch.utils.data import DataLoader
> 
> x_train = torch.FloatTensor([[73, 80, 75],
>                             [93, 88, 93],
>                             [89, 91, 90],
>                             [96, 98, 100],
>                             [73, 66, 70]])
> 
> y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])
> 
> dataset = TensorDataset(x_train, y_train)
> # dataset으로 저장
> 
> dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
> # 통상적으로 2개의 인자를 받는다(데이터셋, 배치크기).
> # suffle=True로 Epoch마다 데이터셋을 섞어서 데이터가 학습되는 순서가 바뀐다
> 
> model = nn.Linear(3, 1)
> optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)
> 
> nb_epochs = 20
> for epoch in range(nb_epochs+1):
>     for batch_idx, samples, in enumerate(dataloader):
>         print(batch_idx)
>         print(samples)
>         
>         x_train, y_train = samples
>         prediction = model(x_train)
>         
>         cost = F.mas_loss(prediction, y_train)
>         
>         optimizer.zero_grad()
>         cost.backward()
>         optimizer.step()
>         
>         print(f'Epoch: {epoch}, Batch: {nb_epochs}, cost: {cost}')
>         
>         
> new_var = torch.FloatTensor([[73, 80, 75]])
> pred_y = model(new_var)
> print("훈련 후 입력이 73, 80, 75일때 예측값: "pred_y)        
> ```

#### Custom Dataset

> torch.utils.data.Dataset를 상속받아 직접 커스텀 데이터셋을 만들 수 있다.
>
> * Custom Dataset의 가장 기본적인 뼈대
>
>   ```python
>   class CustomDataset(torch.utils.data.Dataset):
>       def __init__(self):
>           # 데이터의 전처리를 한다
>       def __len__(self): # len(dataset) 연산
>           # 데이터셋의 길이. 즉 총 생플 수를 적어주는 부분
>       
>       def __getitem__(self, idx): # dataset[i] 연산
>           # 데이터셋에서 특정 1개의 샘플을 가져오는 함수
>   ```
>
> * Custom Dataset으로 선형회귀 구현하기
>
>   ```python
>   import torch
>   import torch.nn.functional as F
>                             
>   from torch.utils.data import Dataset
>   from torch.utils.data import DataLoader
>                             
>   class CustomDataset(Dataset):
>       def __init__(self):
>           self.x_data = [[1, 2, 3],
>                         [4, 5, 6],
>                         [7, 8, 9].
>                         [10, 11, 12]]
>           self.y_data = [[1], [2], [3], [4]]
>                                     
>       def __len__(self):
>           return len(self.x_data)
>                                 
>       def __getitem__(self, idx):
>           x = torch.FloatTensor(self.x_data[idx])
>           y = torch.Floattensor(self.y_data[idx])
>           return (x, y)
>                                 
>   if __name__ == "__main__":
>       dataset = CustomDataset()
>       dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
>                                 
>       model = torch.nn.Linear(3, 1)
>       optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)
>                                 
>       nb_epochs = 20
>       for epoch in range(nb_epochs + 1):
>           for batch_idx, samples, in enumerate(dataloader):
>               x_train, y_train = samples
>                                         
>               prediction = model(x_train)
>               cost = F.mse_loss(prediction, y_train)
>                                         
>               optimizer.zero_grad()
>               cost.backward()
>               optimizer.step()
>                                         
>               print("Epoch: ...")
>                                         
>       new_var = torch.FloatTensor([[73, 80, 75]])
>       pred_y = model(new_var)
>       print("훈련 후 입력이 73, 80, 75일때 예측값: ", pred_y)
>                                 
>   ```

### Logstic Regression

> 두 개의 선택지 중에서 정답을 고르는 문제 - 이진분류
>
> 선형회귀에서의 H(x) = Wx + b가 아니라 S자 모양의 시그모이드 함수를 사용해야 함

#### SIgmoid function

> H(x) = sigmiod(Wx+b) = 1/(1+e^-(Wx+b)) = sigma(Wx + b)
>
> * W와 b의 의미를 알아보자
>
>   ```python
>   import numpy as np   
>   import matplotlib.pyplot as plt   
>
>   def sigmoid(x, w, b):   
>       return 1/(1 + np.exp(-x*w+b))   
>
>   x = np.arange(-5.0, 5.0, 0.1)   
>   y1 = sigmoid(x, 0.5, 0)   
>   y2 = sigmoid(x, 1, 0)    
>   y3 = sigmoid(x, 2, 0)    
>
>   plt.plot(x, y1, "r", linestyle='--')   
>   plt.plot(x, y2, 'g')   
>   plt.plot(x, y3, 'b', linestyle='--')   
>   plt.plot([0, 0], [1.0, 0.0], ":")   
>   plt.title("sigmoid function")   
>   plt.show()   
>   ```
>   ```python
>   import numpy as np   
>   import matplotlib.pyplot as plt   
>
> 	  def sigmoid(x, w, b):   
>       return 1/(1 + np.exp(-x*w+b))   
>
> 	x = np.arange(-5.0, 5.0, 0.1)   
>   y1 = sigmoid(x, 1, -5)   
>   y2 = sigmoid(x, 1, 0)   
>   y3 = sigmoid(x, 1, 5)   
>
> 	plt.plot(x, y1, "r", linestyle='--')   
>   plt.plot(x, y2, 'g')   
>   plt.plot(x, y3, 'b', linestyle='--')   
>   plt.plot([0, 0], [1.0, 0.0], ":")   
>   plt.title("sigmoid function")   
>   plt.show()   
>
>   ```
>
>  * W값이 커질수록 그래프가 급격하게 증가한다
>   * b값에 따라 그래프가 좌우로 이동한다
>
> * sigmoid 함수의 Cost function
>
>  선형회귀에서 사용했던 평균 제곱 오차를 cost function으로 사용할 경우 여러 개의 극소값을 가지게 된다.
>
>  -> cost function으로 log함수를 사용한다
>
> y = 1 -> cost(H(x), y) = -log(H(x))
>
> y = 0 -> cost(H(x), y) = -log(1-H(x))
>
> -> cost(H(x), y) = [ylogH(x) + (1-y)log(1-H(x))]
>
> cost(W) = -1/n * sigma(y_i * logH(x_i) + (1 - y_i)log(1-H(x_i)))

#### Logistic regression 

> ```python
> import torch
> import torch.nn as nn
> import torch.nn.functional as F
> import torch.optim as optim
> 
> x_data = [[1, 2], [2, 3], [3, 4], [4, 3], [5, 3], [6, 2]]
> y_data = [[0], [0], [0], [1], [1], [1]]
> 
> x_train = torch.FLoatTensor(x_data)
> y_train = torch.FloatTensor(y_data)
> 
> W = torch.zeros((2, 1), requires_grad=True)
> b = torch.zeros(1, requries_grad=True)
> 
> optimizer = optim.SGD([W, b], lr=1)
> 
> nb_epoch = 1000
> for epoch in range(nb_epoch):
>     hypothesis = 1 / (1 + torch.exp(-(x_train.matmul(W) + b)))
> 	# hypothesis = torch.sigmoid(x_train.matmul(W) + b)
>     
>     cost = -(y_train * torch.log(hypothesis) + (1-y_train) * torch.log(1 - hypothesis)).mean()
>     # cost = F.binary_cross_entropy(hypothesis, y_train)
> 	
>     optimizer.zeor_grad()
>     cost.backward()
>     optimizer.step()
> 
> x_test = [[10, 20], [20, 30], [3, 40], [64, 36], [54, 34], [63, 22]]
> hypothesis = torch.sigmoid(x_test.matmul(W) + b)
> print(hypothesis) # 값 반환
> ```

#### Logistic regression with nn.Module

> ```python
> import torch
> import torch.nn as nn
> import torch.nn.functional as F
> import torch.optim as optim
> 
> x_data = [[1, 2], [2, 3], [3, 4], [4, 3], [5, 3], [6, 2]]
> y_data = [[0], [0], [0], [1], [1], [1]]
> 
> x_train = torch.FLoatTensor(x_data)
> y_train = torch.FloatTensor(y_data)
> 
> model = nn.Sequential(nn.Linear(2, 1), # input dim = 2, output dim = 1 
>                       nn.Sigmoid()) # 출력은 시그모이드 함수를 거친다
> 
> nb_epochs = 1000
> for epochs in range(nb_epochs):
>     hypothesis = model(x_train)
>     
>     cost = F.binary_cross_entropy(hypothesis, y_train)
>     
>     optimizer.zero_grad()
>     cost.backward()
>     optimizer.step()
>     
>     # prediction = (hypothesus >= torch.FloatTensor([0.5]))
>     # 0.5를 넘으면 True간주
>     # correct_prediction = (prediction.float() == y_train)
>     # 실제값과 일치하면 True로 
>     # accuracy = correct_prediction.sum().item() / len(correct_prediction)
>     # 정확도 계산
>     
> print(list(model.parameters()))
> # W와 b를 출력
> ```

#### Logistic regression with class

> ```python
> import torch
> import torch.nn as nn
> import torch.nn.functional as F
> import torch.optim as optim
> 
> class BinaryClassifier(nn.Module):
>     def __init__(self):
>         super().__init__() # nn.Module의 속성들을 가지고 초기화
>         self.linear = nn.Linear(2, 1)
>         self.sigmoid = nn.Sigmoid()
>         
>     def forward(self, x):
>         return self.sigmoid(self.linear(x))
> 
> if __name__ == "__main__":
>     x_data = [[1, 2], [2, 3], [3, 4], [4, 3], [5, 3], [6, 2]]
> 	y_data = [[0], [0], [0], [1], [1], [1]]
> 
> 	x_train = torch.FLoatTensor(x_data)
> 	y_train = torch.FloatTensor(y_data)
>     
>     model = BinaryClassifier()
>     # 이하동일

### Aritificial Neural Network

#### 머신러닝 모델의 평가

> 전체 데이터를 훈련, 검증, 테스트로 분할하여 사용한다.
>
> * 검증용: 모델의 성능을 조정하기 위한 용도. 과적합이 되고 있는지 판단하거나 하이퍼 파라미터의 조정을 위한 용도.
>
>   * 하이퍼파라미터: 값에 따라서 모델의 성능에 영향을 주는 매개변수. 보통 사용자가 직접 정해줄 수 있는 변수. 학습률, 은닉층 수, 뉴런의 수, 드롭아웃 비율 등이 해당됨
>   * 매개변수: 기계가 훈련을 통해 바꾸는 변수.
>
>   훈련용 데이터로 훈련을 시킨 모델은 검증용 데이터를 사용하여 정확도를 검증하며 하이퍼파라미터를 튜닝한다. 
>
>   이후 모델의 성능을 테스트 하기 위해 테스트 데이터를 사용한다.

#### Sample and feature

> * sample: 하나의 데이터 행
> * feature: 종속변수 y를 예측하기 위한 각각의 독립 변수

#### 혼동행렬 (Confusion Matrix)

> * 정확도: 맞춘 문제수 / 전체 문제 수
>
> * 혼동행렬: 각 열은 예측값을, 각 행은 실제 값을 나타낸다.
>
>   [[TP, FN], [FP, TN]] # True: 정답, False: 오답, Positive: True라고 예상, Negative: False라고 예상

#### 정밀도(Precision)

> True라고 대답한 전체 케이스에 대한 TP(True Positive)의 비율
>
> * precision = TP/(TP + FP)

#### 재현률(Recall)

> 실제 값이 양성인 데이터의 전체 개수에 대해서 TP의 비율. 양성인 데이터 중에서 얼마나 양성인지를 예측(재현)했는지를 나타냄
>
> * 재현률 = TP / (TP + FN)

#### BackPropagation (역전파)

> * 순전파
>
>   데이터는 입력층에서 은닉층 방향으로 향하면서 각 입력에 해당하는 가중치(W)가 곱해지고 결과적으로 가중합(sigma W)으로 계산되어 은닉층 뉴런의 시그모이드 함수의 입력값이 된다.    
>
>   은닉층 뉴런의 시그모이드함수는 은닉층 뉴런의 최종 출력값(H)이 된다.    
>
>   출력값 H는 다시 출력층 뉴런으로 향하며 각각의 값에 해당되는 가중치(W)가 곱해지고 다시 가중합 되어 출력층 뉴런의 시그모이드 함수의 입력값(Z)이 된다    
>
>   Z가 출력층 뉴런에서 시그모이드 함수를 지난 값은 이 인공 신경망이 최종적으로 계산한 출력값이다. 
>
>   예측값과 실제값의 오차를 MSE로 구한다
>
> * 역전파 1단계
>
>   순전파가 입력층에서 출력층으로 향한다면 역전파는 반대로 출력층에서 입력층 방향으로 계산하면서 가증치를 업데이트한다.
>
>   이때 출력층과 은닉층 사이의 가중치를 업데이트 하는 단계를 역전파1단계, 그리고 은닉층 사이의 가중치를 업데이트하는 단계를 역전파 2단계라고 한다.
>
>   각 가중치에 대해 dError/dw_i를 계산한다
>
>   이때 chain rule 에 의해 dError/dw_i = dError/d최종결과 * d최종결과/dz_i * dz_i/dw_i 로 계산할 수 있다
>
>   w_i,updated = w_i - lr* dError/dw_i를 계산한다
>
>   역전파 2단계 역시 같은 방법으로 출력층까지 업데이트하며 나아간다

### CNN (Convolution Neural Network)

> * 합성곱층(Convolution layer)과 풀림층(Pooling layer)으로 구성된다

#### 합성곱 연산 (Convolution operation)

> 합성곱 연산을 통해서 이미지의 특징을 추출하는 역할을 한다.
>
> 커널 또는 필터라는 n x m 크기의 행렬로 높이 x 너비 크기의 이미지를 처음부터 끝까지 겹치며 훑으면서 n x m크기의 겹쳐지는 부분의 각 이미지와 커널의 원소 값을 곱해서 모두 더한 값을 출력하는 것.
>
> * 커널은 일반적으로 3x3 또는 5x5를 이용한다
>
> 데이터중 커널 크기만큼의 부분을 커널에 곱하여 특성맵을 만든다
>
> 한번에 데이터 행렬에서 이동하는 칸의 수를 스트라이드(stride)라고 한다
>
> 패딩(입력 데이터의 가장자리에 0을 채우는 것)으로 특성 맵이 입력보다 작아지는 것을 막을 수 있다.
>
> * 합성곱 신경망에서는 커널 행렬의 원소들이 가중치 역할을 한다
> * 합성곱 신경망의 은닉층을 합성곱층 이라고 한다
> * 편향은 하나의 값만 존재하며 커널이 적용된 결과의 모든 원소에 더해진다
> * 커널의 수와 채널의 수(3차원 텐서에서 z방향 성분 개수)는 같아야 한다.
>
> 



----------------------------------------------------------

# 결론

## 다층 퍼셉트론을 이용하여 만들자

