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
>  numpy, python과 동일
>
>* 2D
>
>  t = torch.FloatTensor([1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [10., 11.,12.])

