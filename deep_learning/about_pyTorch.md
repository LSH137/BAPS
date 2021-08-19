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







