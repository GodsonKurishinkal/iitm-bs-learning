# NumPy Essentials Cheatsheet

## Import

```python
import numpy as np
```

## Array Creation

```python
# From lists
np.array([1, 2, 3])
np.array([[1, 2], [3, 4]])

# Zeros and ones
np.zeros(5)              # [0, 0, 0, 0, 0]
np.zeros((3, 4))         # 3x4 array of zeros
np.ones((2, 3))          # 2x3 array of ones

# Range
np.arange(0, 10, 2)      # [0, 2, 4, 6, 8]
np.linspace(0, 1, 5)     # [0, 0.25, 0.5, 0.75, 1.0]

# Random
np.random.rand(3, 4)     # uniform [0, 1)
np.random.randn(3, 4)    # standard normal
np.random.randint(0, 10, (3, 4))  # random integers

# Identity
np.eye(3)                # 3x3 identity matrix
```

## Array Properties

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])
arr.shape        # (2, 3)
arr.ndim         # 2
arr.size         # 6
arr.dtype        # dtype('int64')
```

## Indexing and Slicing

```python
arr = np.array([0, 1, 2, 3, 4, 5])
arr[0]           # 0
arr[-1]          # 5
arr[1:4]         # [1, 2, 3]
arr[::2]         # [0, 2, 4]

# 2D array
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
arr_2d[0, 1]     # 2
arr_2d[0, :]     # [1, 2, 3]
arr_2d[:, 1]     # [2, 5]
```

## Reshaping

```python
arr = np.arange(12)
arr.reshape(3, 4)        # 3x4 array
arr.reshape(2, 2, 3)     # 3D array
arr.flatten()            # 1D array
arr.ravel()              # 1D array (view)
```

## Mathematical Operations

```python
arr = np.array([1, 2, 3, 4])

# Element-wise
arr + 2          # [3, 4, 5, 6]
arr * 2          # [2, 4, 6, 8]
arr ** 2         # [1, 4, 9, 16]
arr / 2          # [0.5, 1, 1.5, 2]

# Array operations
arr1 + arr2      # element-wise addition
arr1 * arr2      # element-wise multiplication
arr1 @ arr2      # matrix multiplication (1D: dot product)

# Functions
np.sqrt(arr)
np.exp(arr)
np.log(arr)
np.sin(arr)
np.cos(arr)
```

## Statistical Operations

```python
arr = np.array([1, 2, 3, 4, 5])

arr.mean()       # 3.0
arr.median()     # 3.0
arr.std()        # 1.414
arr.var()        # 2.0
arr.sum()        # 15
arr.min()        # 1
arr.max()        # 5
arr.argmin()     # 0 (index of min)
arr.argmax()     # 4 (index of max)

# Axis-specific
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
arr_2d.mean(axis=0)  # [2.5, 3.5, 4.5] (column means)
arr_2d.mean(axis=1)  # [2, 5] (row means)
```

## Boolean Indexing

```python
arr = np.array([1, 2, 3, 4, 5])

# Conditions
arr > 3          # [False, False, False, True, True]
arr[arr > 3]     # [4, 5]

# Multiple conditions
arr[(arr > 2) & (arr < 5)]  # [3, 4]
arr[(arr < 2) | (arr > 4)]  # [1, 5]
```

## Linear Algebra

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Matrix multiplication
A @ B
np.dot(A, B)
np.matmul(A, B)

# Transpose
A.T

# Inverse
np.linalg.inv(A)

# Determinant
np.linalg.det(A)

# Eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)

# Solve Ax = b
b = np.array([1, 2])
x = np.linalg.solve(A, b)
```

## Broadcasting

```python
# Scalar with array
arr = np.array([1, 2, 3])
arr + 5          # [6, 7, 8]

# Arrays of different shapes
arr1 = np.array([[1, 2, 3]])     # (1, 3)
arr2 = np.array([[1], [2], [3]]) # (3, 1)
result = arr1 + arr2              # (3, 3)
```

## Useful Functions

```python
# Sorting
np.sort(arr)
np.argsort(arr)      # indices that would sort

# Unique
np.unique(arr)

# Concatenate
np.concatenate([arr1, arr2])
np.vstack([arr1, arr2])  # vertical stack
np.hstack([arr1, arr2])  # horizontal stack

# Split
np.split(arr, 3)         # split into 3 parts

# Where
np.where(arr > 3, 1, 0)  # 1 if true, 0 if false
```

## Common Patterns

```python
# Create meshgrid
x = np.linspace(0, 1, 5)
y = np.linspace(0, 1, 3)
X, Y = np.meshgrid(x, y)

# Apply function to each element
def square(x):
    return x ** 2
vectorized_square = np.vectorize(square)
result = vectorized_square(arr)

# Or use broadcasting directly
result = arr ** 2
```
