import torch
t = torch.FloatTensor([[1., 2., 3.],
                    [4., 5., 6.],
                    [7., 8., 9.],
                    [10., 11., 12.]
                    ])

print(t)
print(f"rank = {t.dim()}")
print(f"shape = {t.size()}")

# 슬라이싱
print("==================== slicing ====================")
print(t[:, 1]) # 첫 번째 차원을 전체 선택한 상황에서 두 번째 차원의 첫 번째 것만 가져온다
# -> tensor([ 2.,  5.,  8., 11.])
print(t[:, 2])
# -> tensor([ 3.,  6.,  9., 12.])
print(t[0:2, 1])
# -> tensor([2., 5.])
print(t[0:3, 0:2])
# tensor([[1., 2.],
#         [4., 5.],
#         [7., 8.]])

# Broadcasting
print("==================== Broadcasting ====================")
