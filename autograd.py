import numpy as np
import torch
import matplotlib.pyplot as plt
import math

"""
Create a numpy array full of 100 values ranging from [-pi, pi]
"""
x_npy = []
for count in range(100):
  x_npy.append(np.random.uniform(-(np.pi), np.pi))
  
print(f'-pi: {-np.pi}, min: {np.pi}')
print(f'min: {min(x_npy)}, max: {max(x_npy)}')
x_npy = np.array(x_npy)

"""
Turn warnings off
"""
import warnings
# Settings the warnings to be ignored
warnings.filterwarnings('ignore')

x = torch.tensor(x_npy)
x.to('cuda')    # We move our tensor to the GPU if available
x.requires_grad_(True)
y = sin_taylor(x)
y.requires_grad_(True)
y.sum().backward()

"""
Get dz/dx and convert to a numpy array
dzdx: gradient of loss (y) with respect to x
- in other words, the result of backpropagation used to update models hyperparameters
"""
dzdx = x.grad
dzdx = dzdx.numpy()
print(f"dzdx: {dzdx}")

"""
Plot dzdx vs. x_npy
proof that the gradient of y is indeed equal to cos(x)
proof dsinx / dx = cosx
"""

"""
Initialize values for cosine graph
"""
x_cos = np.arange(-(np.pi), np.pi, .05)
y_cos = np.cos(x_cos)
# plot overlayed graph
x_val,y_val = x_npy, dzdx, 
plt.scatter(x_val,y_val,)  # dzdx graph in blue
plt.plot(x_cos, y_cos, 'r') # cosine graph in red 
#set title and x, y - axes labels
plt.title('dzdx vs. x_npy')
plt.xlabel('x_npy')
plt.ylabel('dzdx')


