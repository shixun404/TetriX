import torch as th

a = [[1, 1, 0, 1],
     [1, 1, 1, 0],
     [0, 1, 1, 1],
     [1, 0, 1, 1]]

# b = th.as_tensor(a, dtype=th.float)
# b = b / 3
b = th.ones(4, 4) / 4
b_res = th.eye(4)
print(b_res)
for i in range(10):
    b_res = b_res @ b
    print(i, b_res)