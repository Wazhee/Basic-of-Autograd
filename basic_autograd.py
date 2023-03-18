import numpy as np
import math
import torch
import matplotlib.pyplot as plt

"""
Taylor approximation to sin(x). 
x: (float) Input value
n: (int) Number of terms in Taylor approximation
"""

def sin_taylor(x, n=10):
  sum = 0
  for itr in range(n):
    val = (-1)**itr * (x**(2*itr+1)) / np.math.factorial(2*itr + 1)
    sum += val
  return sum # Replace with your code for 1.1a.

# Proof sin_taylor() works
print(sin_taylor(-0.7706)
      
"""
Begining of autograd
Create a tensor with a value of pi/4
"""
x = torch.tensor(np.pi/4, requires_grad=True)
y = sin_taylor(x)
y.requires_grad_(True)
y.backward()
print(f"sin(x): {y}, x.grad: {x.grad}, cos(x): {math.cos(x)}")
      
