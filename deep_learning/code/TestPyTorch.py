import torch
import torchvision
import numpy as np


if __name__ == "__main__":
    data = [[1.0, 2.0], [3.0, 4.0]]
    x_data = torch.tensor(data)

    np_array = np.array(data)
    x_np = torch.from_numpy(np_array)

    x_ones = torch.ones_like(x_data) # x_data와 같은 모양과 자료형의 1로 채워진 텐서를 반환
    print(f"Ones Tensor: \n {x_ones} \n")

    x_rand = torch.rand_like(x_data, dtype=torch.float) #x_data와 같은 모양의 난수로 채워진 텐서를 반환
    print(f"Random Tensor: \n {x_rand} \n")

    shape = (2,3,)
    zeros_tesor = torch.zeros(shape)
    print(f"zeros tensor: \n {zeros_tesor}\n")

    # Attribute of Tensor
    tensor = torch.rand(3, 4)
    print(f"Shape of tensor: {tensor.shape}")
    print(f"Datatype of tensor:{tensor.dtype}")
    print(f"Device tensor is stored on: {tensor.device}")

    # operation of tensor
    if torch.cuda.is_available(): # GPU가 존재하면 더 빠르게 연산 가능
        tensor = tensor.to("cuda")

    print(f"Device tensor is stored on: {tensor.device}")

    # slicing
    tensor = torch.ones(4, 4)
    tensor[:, 1] = 2 # 모든 행의 인덱스가 1인 요소 꺼내기, tensor[2, :] -> 2행의 모든 요소 꺼내기
    print(tensor)

    # linking tensor
    t1 = torch.cat([tensor, tensor, tensor], dim=1)
    print(t1)
    ## another tensor plus: torch.stack
    ## in-place operation
    print(tensor, "\n")
    tensor.add_(5)
    print(tensor)
    # in-place 연산은 메모리 일부를 절약하지만, 기록이 즉시 삭제되어 도함수 계산에 문제가 발생할 수 있다. 따라서 사용을 권장하지 않는다.

    # multiply tensor
    ## element-wise product
    print(f"tensor.mul(tensor)\n {tensor.mul(tensor)}\n")
    ## is same as
    print(f"tensor * tensor = \n{tensor * tensor}\n")

    ## matrix multiplication
    print(f"tensor.matmul(tensor.T) = \n {tensor.matmul(tensor.T)}\n")
    ## is same as
    print(f"tensor @ tensor.T = \n {tensor @ tensor.T}\n")

    # NumPy 변환
    ## cpu상의 텐서와 NumPy 배열은 메모리 공간을 공유하기 때문에 하나를 변경하면 다른 하나도 변경됩니다.
    t = torch.ones(5)
    print(f"t: {t}")
    n = t.numpy()
    print(f"n: {n}")

    t.add(1)
    print(f"t: {t}")
    print(f"n: {n}")

    # tensor의 변경사항이 numpy 배열에도 반영된다.
    # 거꾸로 numpy배열의 변경사항도 tensor에 반영된다.