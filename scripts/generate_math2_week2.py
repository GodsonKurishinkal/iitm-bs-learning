#!/usr/bin/env python3
"""
Generate Mathematics-II Week 2: Matrix Operations
Comprehensive notebook with theory, code, and visualizations
"""
import json

def create_notebook():
    nb = {
        "cells": [],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python", "version": "3.9.6"}},
        "nbformat": 4, "nbformat_minor": 5
    }
    
    cells = [
        # Cell 1: Title
        {"cell_type": "markdown", "metadata": {}, "source": [
            "# Week 2: Matrix Operations\n\n",
            "**Course:** Mathematics for Data Science II (BSMA1003)  \n",
            "**Level:** Foundation  \n",
            "**Week:** 2\n\n",
            "## Topics Covered\n",
            "1. Matrix addition and scalar multiplication\n",
            "2. Matrix multiplication and properties\n",
            "3. Matrix transpose and symmetric matrices\n",
            "4. Matrix inverse and invertibility\n",
            "5. Special matrices (identity, diagonal, triangular)\n\n",
            "## Learning Objectives\n",
            "- Perform matrix operations efficiently using NumPy\n",
            "- Understand non-commutative nature of matrix multiplication\n",
            "- Calculate matrix inverses and solve linear equations\n",
            "- Recognize special matrix patterns\n",
            "- Apply matrix operations to data science problems\n"
        ]},
        
        # Cell 2: Imports
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "from scipy import linalg\n\n",
            "np.random.seed(42)\n",
            "plt.style.use('seaborn-v0_8-whitegrid')\n",
            "%matplotlib inline\n\n",
            "print('✓ Libraries loaded')"
        ]},
        
        # Cell 3: Theory - Matrix Addition
        {"cell_type": "markdown", "metadata": {}, "source": [
            "## 1. Matrix Addition and Scalar Multiplication\n\n",
            "### Matrix Addition\n",
            "Two matrices can be added if they have the **same dimensions**.\n\n",
            "$$A + B = \\begin{bmatrix} a_{11} & a_{12} \\\\ a_{21} & a_{22} \\end{bmatrix} + \\begin{bmatrix} b_{11} & b_{12} \\\\ b_{21} & b_{22} \\end{bmatrix} = \\begin{bmatrix} a_{11}+b_{11} & a_{12}+b_{12} \\\\ a_{21}+b_{21} & a_{22}+b_{22} \\end{bmatrix}$$\n\n",
            "**Properties:**\n",
            "- **Commutative:** $A + B = B + A$\n",
            "- **Associative:** $(A + B) + C = A + (B + C)$\n",
            "- **Zero matrix:** $A + 0 = A$\n\n",
            "### Scalar Multiplication\n",
            "Multiply every element by a scalar $k$:\n\n",
            "$$kA = k\\begin{bmatrix} a_{11} & a_{12} \\\\ a_{21} & a_{22} \\end{bmatrix} = \\begin{bmatrix} ka_{11} & ka_{12} \\\\ ka_{21} & ka_{22} \\end{bmatrix}$$\n\n",
            "**Properties:**\n",
            "- $k(A + B) = kA + kB$\n",
            "- $(k + m)A = kA + mA$\n",
            "- $k(mA) = (km)A$\n"
        ]},
        
        # Cell 4: Matrix Addition Code
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
            "# Matrix addition and scalar multiplication\n",
            "A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])\n",
            "B = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]])\n",
            "k = 3\n\n",
            "print('Matrix A:')\n",
            "print(A)\n",
            "print('\\nMatrix B:')\n",
            "print(B)\n",
            "print('\\nA + B:')\n",
            "print(A + B)\n",
            "print('\\n3A:')\n",
            "print(k * A)\n",
            "print('\\n2A + B:')\n",
            "print(2*A + B)\n\n",
            "# Verify commutativity\n",
            "print('\\n✓ A + B = B + A:', np.array_equal(A + B, B + A))"
        ]},
        
        # Cell 5: Theory - Matrix Multiplication
        {"cell_type": "markdown", "metadata": {}, "source": [
            "## 2. Matrix Multiplication\n\n",
            "### Definition\n",
            "For matrices $A_{m \\times n}$ and $B_{n \\times p}$, the product $AB$ is an $m \\times p$ matrix where:\n\n",
            "$$(AB)_{ij} = \\sum_{k=1}^{n} a_{ik}b_{kj}$$\n\n",
            "**Key Point:** Number of columns in $A$ must equal number of rows in $B$!\n\n",
            "### Example (2×3) × (3×2)\n",
            "$$\\begin{bmatrix} a_{11} & a_{12} & a_{13} \\\\ a_{21} & a_{22} & a_{23} \\end{bmatrix} \\begin{bmatrix} b_{11} & b_{12} \\\\ b_{21} & b_{22} \\\\ b_{31} & b_{32} \\end{bmatrix} = \\begin{bmatrix} c_{11} & c_{12} \\\\ c_{21} & c_{22} \\end{bmatrix}$$\n\n",
            "where $c_{11} = a_{11}b_{11} + a_{12}b_{21} + a_{13}b_{31}$\n\n",
            "### Properties\n",
            "- **NOT commutative:** $AB \\neq BA$ (in general)\n",
            "- **Associative:** $(AB)C = A(BC)$\n",
            "- **Distributive:** $A(B + C) = AB + AC$\n",
            "- **Identity:** $AI = IA = A$ where $I$ is identity matrix\n\n",
            "### Data Science Connection\n",
            "Matrix multiplication is the foundation of:\n",
            "- Neural network forward propagation\n",
            "- Data transformations\n",
            "- Graph algorithms\n",
            "- Recommendation systems\n"
        ]},
        
        # Cell 6: Matrix Multiplication Code
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
            "# Matrix multiplication\n",
            "A = np.array([[1, 2], [3, 4], [5, 6]])  # 3×2\n",
            "B = np.array([[7, 8, 9], [10, 11, 12]])  # 2×3\n\n",
            "print('A (3×2):')\n",
            "print(A)\n",
            "print('\\nB (2×3):')\n",
            "print(B)\n",
            "print('\\nAB (3×3):')\n",
            "AB = A @ B  # or np.dot(A, B)\n",
            "print(AB)\n",
            "print('\\nBA (2×2):')\n",
            "BA = B @ A\n",
            "print(BA)\n\n",
            "# Demonstrate non-commutativity with square matrices\n",
            "C = np.array([[1, 2], [3, 4]])\n",
            "D = np.array([[5, 6], [7, 8]])\n",
            "print('\\nC:')\n",
            "print(C)\n",
            "print('\\nD:')\n",
            "print(D)\n",
            "print('\\nCD:')\n",
            "print(C @ D)\n",
            "print('\\nDC:')\n",
            "print(D @ C)\n",
            "print('\\n✗ CD ≠ DC:', not np.array_equal(C @ D, D @ C))"
        ]},
        
        # Cell 7: Visualization
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
            "# Visualize matrix multiplication\n",
            "fig, axes = plt.subplots(2, 2, figsize=(12, 10))\n\n",
            "# Original matrices\n",
            "im1 = axes[0, 0].imshow(C, cmap='Blues', aspect='auto')\n",
            "axes[0, 0].set_title('Matrix C', fontsize=14, fontweight='bold')\n",
            "for i in range(2):\n",
            "    for j in range(2):\n",
            "        axes[0, 0].text(j, i, f'{C[i,j]}', ha='center', va='center', fontsize=16, fontweight='bold')\n",
            "plt.colorbar(im1, ax=axes[0, 0])\n\n",
            "im2 = axes[0, 1].imshow(D, cmap='Greens', aspect='auto')\n",
            "axes[0, 1].set_title('Matrix D', fontsize=14, fontweight='bold')\n",
            "for i in range(2):\n",
            "    for j in range(2):\n",
            "        axes[0, 1].text(j, i, f'{D[i,j]}', ha='center', va='center', fontsize=16, fontweight='bold')\n",
            "plt.colorbar(im2, ax=axes[0, 1])\n\n",
            "# Products\n",
            "CD = C @ D\n",
            "im3 = axes[1, 0].imshow(CD, cmap='Reds', aspect='auto')\n",
            "axes[1, 0].set_title('CD (Different)', fontsize=14, fontweight='bold')\n",
            "for i in range(2):\n",
            "    for j in range(2):\n",
            "        axes[1, 0].text(j, i, f'{CD[i,j]}', ha='center', va='center', fontsize=16, fontweight='bold')\n",
            "plt.colorbar(im3, ax=axes[1, 0])\n\n",
            "DC = D @ C\n",
            "im4 = axes[1, 1].imshow(DC, cmap='Purples', aspect='auto')\n",
            "axes[1, 1].set_title('DC (Different)', fontsize=14, fontweight='bold')\n",
            "for i in range(2):\n",
            "    for j in range(2):\n",
            "        axes[1, 1].text(j, i, f'{DC[i,j]}', ha='center', va='center', fontsize=16, fontweight='bold')\n",
            "plt.colorbar(im4, ax=axes[1, 1])\n\n",
            "plt.suptitle('Matrix Multiplication is NOT Commutative: CD ≠ DC', fontsize=16, fontweight='bold')\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]},
        
        # Cell 8: Theory - Transpose
        {"cell_type": "markdown", "metadata": {}, "source": [
            "## 3. Matrix Transpose\n\n",
            "### Definition\n",
            "The **transpose** of matrix $A$ (denoted $A^T$) is obtained by swapping rows and columns:\n\n",
            "$$A = \\begin{bmatrix} a_{11} & a_{12} & a_{13} \\\\ a_{21} & a_{22} & a_{23} \\end{bmatrix} \\implies A^T = \\begin{bmatrix} a_{11} & a_{21} \\\\ a_{12} & a_{22} \\\\ a_{13} & a_{23} \\end{bmatrix}$$\n\n",
            "**Properties:**\n",
            "- $(A^T)^T = A$\n",
            "- $(A + B)^T = A^T + B^T$\n",
            "- $(kA)^T = kA^T$\n",
            "- $(AB)^T = B^TA^T$ (order reverses!)\n\n",
            "### Symmetric Matrices\n",
            "A matrix is **symmetric** if $A = A^T$\n\n",
            "Example: $\\begin{bmatrix} 1 & 2 & 3 \\\\ 2 & 4 & 5 \\\\ 3 & 5 & 6 \\end{bmatrix}$\n\n",
            "**Important:** Covariance matrices in statistics are always symmetric!\n"
        ]},
        
        # Cell 9: Transpose Code
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
            "# Matrix transpose\n",
            "A = np.array([[1, 2, 3], [4, 5, 6]])\n",
            "print('Matrix A (2×3):')\n",
            "print(A)\n",
            "print('\\nTranspose A^T (3×2):')\n",
            "print(A.T)\n\n",
            "# Create symmetric matrix\n",
            "B = np.array([[1, 2, 3], [2, 4, 5], [3, 5, 6]])\n",
            "print('\\nSymmetric Matrix B:')\n",
            "print(B)\n",
            "print('\\nB^T:')\n",
            "print(B.T)\n",
            "print('\\n✓ B is symmetric:', np.array_equal(B, B.T))\n\n",
            "# Verify (AB)^T = B^T A^T\n",
            "X = np.array([[1, 2], [3, 4]])\n",
            "Y = np.array([[5, 6], [7, 8]])\n",
            "XY = X @ Y\n",
            "print('\\nVerify (XY)^T = Y^T X^T:')\n",
            "print('(XY)^T:')\n",
            "print(XY.T)\n",
            "print('\\nY^T X^T:')\n",
            "print(Y.T @ X.T)\n",
            "print('\\n✓ Equal:', np.allclose(XY.T, Y.T @ X.T))"
        ]},
        
        # Cell 10: Theory - Matrix Inverse
        {"cell_type": "markdown", "metadata": {}, "source": [
            "## 4. Matrix Inverse\n\n",
            "### Definition\n",
            "For a square matrix $A$, its **inverse** $A^{-1}$ satisfies:\n\n",
            "$$AA^{-1} = A^{-1}A = I$$\n\n",
            "where $I$ is the identity matrix.\n\n",
            "### 2×2 Matrix Inverse Formula\n",
            "$$A = \\begin{bmatrix} a & b \\\\ c & d \\end{bmatrix} \\implies A^{-1} = \\frac{1}{ad-bc}\\begin{bmatrix} d & -b \\\\ -c & a \\end{bmatrix}$$\n\n",
            "**Requirements:**\n",
            "- Matrix must be **square** ($n \\times n$)\n",
            "- **Determinant** $\\det(A) \\neq 0$ (non-singular)\n\n",
            "### Properties\n",
            "- $(A^{-1})^{-1} = A$\n",
            "- $(AB)^{-1} = B^{-1}A^{-1}$ (order reverses!)\n",
            "- $(A^T)^{-1} = (A^{-1})^T$\n",
            "- $(kA)^{-1} = \\frac{1}{k}A^{-1}$\n\n",
            "### Applications\n",
            "- Solving linear systems: $Ax = b \\implies x = A^{-1}b$\n",
            "- Data transformations and reversals\n",
            "- Cryptography\n",
            "- Computer graphics transformations\n"
        ]},
        
        # Cell 11: Matrix Inverse Code
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
            "# Matrix inverse\n",
            "A = np.array([[4, 7], [2, 6]])\n",
            "print('Matrix A:')\n",
            "print(A)\n\n",
            "# Calculate determinant\n",
            "det_A = np.linalg.det(A)\n",
            "print(f'\\nDeterminant: {det_A:.2f}')\n\n",
            "# Calculate inverse\n",
            "if det_A != 0:\n",
            "    A_inv = np.linalg.inv(A)\n",
            "    print('\\nInverse A^(-1):')\n",
            "    print(A_inv)\n",
            "    \n",
            "    # Verify AA^(-1) = I\n",
            "    I = A @ A_inv\n",
            "    print('\\nAA^(-1):')\n",
            "    print(I)\n",
            "    print('\\n✓ AA^(-1) = I:', np.allclose(I, np.eye(2)))\n",
            "    \n",
            "    # Solve linear system Ax = b\n",
            "    b = np.array([1, 2])\n",
            "    x = A_inv @ b\n",
            "    print(f'\\nSolve Ax = b where b = {b}')\n",
            "    print(f'Solution x = A^(-1)b = {x}')\n",
            "    print(f'Verification Ax = {A @ x}')\n",
            "else:\n",
            "    print('\\n✗ Matrix is singular (not invertible)')\n\n",
            "# Example of singular matrix\n",
            "B = np.array([[1, 2], [2, 4]])  # Rows are proportional\n",
            "print('\\n\\nSingular Matrix B:')\n",
            "print(B)\n",
            "print(f'Determinant: {np.linalg.det(B):.10f}')\n",
            "print('✗ Not invertible (det = 0)')"
        ]},
        
        # Cell 12: Theory - Special Matrices
        {"cell_type": "markdown", "metadata": {}, "source": [
            "## 5. Special Matrices\n\n",
            "### Identity Matrix $I$\n",
            "$$I_3 = \\begin{bmatrix} 1 & 0 & 0 \\\\ 0 & 1 & 0 \\\\ 0 & 0 & 1 \\end{bmatrix}$$\n",
            "Property: $AI = IA = A$\n\n",
            "### Diagonal Matrix $D$\n",
            "$$D = \\begin{bmatrix} d_1 & 0 & 0 \\\\ 0 & d_2 & 0 \\\\ 0 & 0 & d_3 \\end{bmatrix}$$\n",
            "All off-diagonal elements are zero.\n\n",
            "### Upper Triangular Matrix $U$\n",
            "$$U = \\begin{bmatrix} u_{11} & u_{12} & u_{13} \\\\ 0 & u_{22} & u_{23} \\\\ 0 & 0 & u_{33} \\end{bmatrix}$$\n",
            "All elements below diagonal are zero.\n\n",
            "### Lower Triangular Matrix $L$\n",
            "$$L = \\begin{bmatrix} l_{11} & 0 & 0 \\\\ l_{21} & l_{22} & 0 \\\\ l_{31} & l_{32} & l_{33} \\end{bmatrix}$$\n",
            "All elements above diagonal are zero.\n\n",
            "### Zero Matrix $0$\n",
            "All elements are zero. Property: $A + 0 = A$\n\n",
            "### Properties\n",
            "- Product of diagonal matrices is diagonal\n",
            "- Inverse of diagonal matrix is diagonal (if all $d_i \\neq 0$)\n",
            "- Triangular matrices important in LU decomposition\n"
        ]},
        
        # Cell 13: Special Matrices Code
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
            "# Special matrices\n",
            "n = 4\n\n",
            "# Identity matrix\n",
            "I = np.eye(n)\n",
            "print('Identity Matrix I:')\n",
            "print(I)\n\n",
            "# Diagonal matrix\n",
            "D = np.diag([2, 3, 5, 7])\n",
            "print('\\nDiagonal Matrix D:')\n",
            "print(D)\n\n",
            "# Upper triangular\n",
            "U = np.array([[1, 2, 3, 4], [0, 5, 6, 7], [0, 0, 8, 9], [0, 0, 0, 10]])\n",
            "print('\\nUpper Triangular U:')\n",
            "print(U)\n\n",
            "# Lower triangular\n",
            "L = np.array([[1, 0, 0, 0], [2, 3, 0, 0], [4, 5, 6, 0], [7, 8, 9, 10]])\n",
            "print('\\nLower Triangular L:')\n",
            "print(L)\n\n",
            "# Zero matrix\n",
            "Z = np.zeros((3, 3))\n",
            "print('\\nZero Matrix:')\n",
            "print(Z)\n\n",
            "# Properties demonstration\n",
            "A = np.random.randint(1, 10, (4, 4))\n",
            "print('\\n\\nTest Matrix A:')\n",
            "print(A)\n",
            "print('\\nAI (should equal A):')\n",
            "print(A @ I)\n",
            "print('\\n✓ AI = A:', np.array_equal(A @ I, A))"
        ]},
        
        # Cell 14: Real-World Application
        {"cell_type": "markdown", "metadata": {}, "source": [
            "## 6. Real-World Application: Image Transformation\n\n",
            "### Problem\n",
            "Digital images can be transformed using matrix operations:\n",
            "- Rotation\n",
            "- Scaling\n",
            "- Shearing\n",
            "- Brightness adjustment (scalar multiplication)\n\n",
            "### Rotation Matrix\n",
            "Rotate by angle $\\theta$ counterclockwise:\n",
            "$$R(\\theta) = \\begin{bmatrix} \\cos\\theta & -\\sin\\theta \\\\ \\sin\\theta & \\cos\\theta \\end{bmatrix}$$\n\n",
            "### Scaling Matrix\n",
            "Scale by factors $s_x, s_y$:\n",
            "$$S = \\begin{bmatrix} s_x & 0 \\\\ 0 & s_y \\end{bmatrix}$$\n"
        ]},
        
        # Cell 15: Image Transformation Code
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
            "# Create simple image (matrix)\n",
            "image = np.array([\n",
            "    [100, 100, 100, 100, 100],\n",
            "    [100, 200, 200, 200, 100],\n",
            "    [100, 200, 255, 200, 100],\n",
            "    [100, 200, 200, 200, 100],\n",
            "    [100, 100, 100, 100, 100]\n",
            "])\n\n",
            "# Operations\n",
            "dark_image = 0.5 * image  # Darken (scalar multiplication)\n",
            "bright_image = 1.5 * image  # Brighten\n",
            "bright_image = np.clip(bright_image, 0, 255)  # Keep in valid range\n\n",
            "# Visualize\n",
            "fig, axes = plt.subplots(1, 3, figsize=(15, 5))\n\n",
            "axes[0].imshow(image, cmap='gray', vmin=0, vmax=255)\n",
            "axes[0].set_title('Original Image', fontsize=14, fontweight='bold')\n",
            "axes[0].axis('off')\n\n",
            "axes[1].imshow(dark_image, cmap='gray', vmin=0, vmax=255)\n",
            "axes[1].set_title('Darkened (0.5 × Image)', fontsize=14, fontweight='bold')\n",
            "axes[1].axis('off')\n\n",
            "axes[2].imshow(bright_image, cmap='gray', vmin=0, vmax=255)\n",
            "axes[2].set_title('Brightened (1.5 × Image)', fontsize=14, fontweight='bold')\n",
            "axes[2].axis('off')\n\n",
            "plt.suptitle('Image Brightness Adjustment via Scalar Multiplication', \n",
            "             fontsize=16, fontweight='bold')\n",
            "plt.tight_layout()\n",
            "plt.show()\n\n",
            "print('Application: Photo editing, computer vision, data augmentation in ML')"
        ]},
        
        # Cell 16: Summary
        {"cell_type": "markdown", "metadata": {}, "source": [
            "## Summary & Key Takeaways\n\n",
            "### Core Operations\n",
            "1. **Addition/Scalar Multiplication**\n",
            "   - Same dimensions required\n",
            "   - Commutative and associative\n",
            "   - Element-wise operations\n\n",
            "2. **Matrix Multiplication**\n",
            "   - Dimensions: $(m \\times n)(n \\times p) = (m \\times p)$\n",
            "   - **NOT commutative:** $AB \\neq BA$\n",
            "   - Foundation of ML and data transformations\n\n",
            "3. **Transpose**\n",
            "   - Swap rows and columns\n",
            "   - $(AB)^T = B^TA^T$ (order reverses!)\n",
            "   - Symmetric matrices: $A = A^T$\n\n",
            "4. **Inverse**\n",
            "   - Only for square matrices with $\\det(A) \\neq 0$\n",
            "   - $AA^{-1} = I$\n",
            "   - Used to solve linear systems\n\n",
            "5. **Special Matrices**\n",
            "   - Identity: $AI = A$\n",
            "   - Diagonal: Fast computation\n",
            "   - Triangular: Used in decompositions\n\n",
            "### Data Science Applications\n",
            "- **Neural Networks:** Forward/backward propagation\n",
            "- **Computer Vision:** Image transformations\n",
            "- **Statistics:** Covariance matrices (symmetric)\n",
            "- **Linear Regression:** Normal equations use inverse\n",
            "- **Dimensionality Reduction:** PCA uses matrix operations\n\n",
            "### Important Formulas\n",
            "$$A + B \\quad kA \\quad AB \\quad A^T \\quad A^{-1}$$\n",
            "$$\\det(A) \\neq 0 \\iff A \\text{ invertible}$$\n",
            "$$(AB)^T = B^TA^T \\quad (AB)^{-1} = B^{-1}A^{-1}$$\n\n",
            "### Next Steps\n",
            "- Week 3: Solving linear equation systems\n",
            "- Week 4: Determinants in depth\n",
            "- Practice: Implement matrix operations from scratch\n"
        ]}
    ]
    
    nb['cells'] = cells
    return nb

if __name__ == '__main__':
    nb = create_notebook()
    path = '../01-Foundation-Level/01-Mathematics-II/notebooks/week-02-matrix-operations.ipynb'
    with open(path, 'w') as f:
        json.dump(nb, f, indent=2)
    print(f'✓ Week 2: {len(nb["cells"])} cells')
