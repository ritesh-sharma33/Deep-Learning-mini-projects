import numpy as np

def softmax(L):
  expL = np.exp(L)
  sumExpL = sum(expL)
  result = []
  for i in expL:
    result.append(i * 1.0 / sumExpL)
  return result

## We can also use np.divide(expL, expL.sum())

result = softmax([2, 1, 0])
print(result)