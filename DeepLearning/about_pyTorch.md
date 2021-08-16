# PyTorch

***

official reference: https://tutorials.pytorch.kr/beginner/blitz/tensor_tutorial.html   

## Tensor

> 배열이나 행렬과 매우 유사한 특수한 자료구조. PyTorch에서는 텐서를 사용하여 모델의 입력과 출력, 그리고 모델의 매개변수들을 인코딩 한다.   
>
> 텐서는 GPU나 다른 하드웨어 가속기에서 실행할 수 있다는 점만 제외하면 NumPy의 ndarray와 유사하다.

> > <pre>
> >     <code>
> >     import torch
> >     import numpy as np
> >     
> >     # tenser 초기화
> >     # 데이터로부터 직접 생성하기. 데이터의 자료형을 자동으로 유추
> >     data = [[1, 2], [3, 4]]
> >     x_data = torch.tensor(data)
> >     
> >     # Numpy 배열로부터 생성하기
> >     np_array = ap.array(data)
> >     x_np = torch.from_numpy(np_array)
> >     
> >     # 다른 텐서로부터 생성하기
> >     x_ones = torch.ones_like(x_data) # x_data의 속성을 유지한다
> >     print(f"Ones Tensor: \n {x_ones} \n")
> >     
> >     x_rand = torch.rand_like(x_data, dtype=torch.float) # x_data의 속성을 덮어쓴다
> >     print(f"Random Tensor: \n {x_rand} \n)
> >     
> >     
> >     </code>
> > </pre>
> >
> > 

