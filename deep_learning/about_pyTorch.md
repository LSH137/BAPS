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
>  print(m1 + m2) -> tensor([[5., 5.]])   
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

>t = torch.FLoatTensor([[1, 2], [3, 4]])   
>
>print(t.mean()) -> tensor(2.5000)   
>
>t.mean(dim=0) -> 입력에서 첫 번째 차원을 제거한다 -> 결과가 행이 제거되고 열만 있는 벡터
>
>1과 3의 평균을 구하고, 2와 4의 평균을 구한다 -> [2., 3.]
>
>t.mean(dim=1) -> 입력에서 두 번째 차원을 제거한다 -> 결과가 열이 제거되고 행만 있는 벡터
>
>t.mean(dim=-1) : 마지막 차원을 결과에서 제거

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

