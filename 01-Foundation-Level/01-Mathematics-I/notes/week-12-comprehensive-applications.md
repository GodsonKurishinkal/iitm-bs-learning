# Week 12: Comprehensive Applications - Synthesis and Integration

---
**Date**: 2025-11-22
**Course**: BSMA1001 - Mathematics for Data Science I
**Level**: Foundation
**Week**: 12 of 12
**Source**: IIT Madras Mathematics I Week 12
**Topic Area**: Applied Mathematics - Data Science Integration
**Tags**: #BSMA1001 #Applications #Optimization #MachineLearning #Week12 #Foundation
---

## Overview

This final week synthesizes **all concepts from Weeks 4-11** and demonstrates their applications to data science and machine learning. We bridge the gap between mathematical theory and practical implementation, showing how calculus, functions, and series form the foundation of modern AI algorithms.

**Comprehensive Review Path (Weeks 4-11):**
1. **Week 4:** Polynomials and algebraic foundations
2. **Week 5:** Functions (domain, range, composition, inverses)
3. **Week 6:** Exponential and logarithmic functions
4. **Week 7:** Trigonometric functions and identities
5. **Week 8:** Sequences, series, convergence
6. **Week 9:** Limits and continuity
7. **Week 10:** Derivatives and optimization
8. **Week 11:** Integration and probability

**This Week's Focus:**
- Multivariable calculus preview (gradients, Hessians)
- Optimization algorithms in machine learning
- Taylor series for function approximation
- Differential equations basics
- End-to-end ML applications combining all concepts

---

## 1. From Single-Variable to Multivariable Calculus

### 1.1 Motivation: Loss Functions in Machine Learning

In linear regression, we minimize a loss function of **multiple** parameters:

$$L(w_0, w_1, \ldots, w_n) = \frac{1}{2m} \sum_{i=1}^{m} (h_w(x^{(i)}) - y^{(i)})^2$$

where:
- $w_0, w_1, \ldots, w_n$ are model parameters (weights)
- $h_w(x) = w_0 + w_1 x_1 + \cdots + w_n x_n$ is the hypothesis function
- $m$ is the number of training examples

To minimize $L$, we need derivatives with respect to **each** parameter → **partial derivatives**.

### 1.2 Partial Derivatives

For a function $f(x, y)$ of two variables:
- **Partial derivative with respect to x:** $\frac{\partial f}{\partial x}$ (treat $y$ as constant)
- **Partial derivative with respect to y:** $\frac{\partial f}{\partial y}$ (treat $x$ as constant)

**Example:** $f(x, y) = x^2y + 3xy^2$

$$\frac{\partial f}{\partial x} = 2xy + 3y^2 \quad \text{(treat y as constant)}$$
$$\frac{\partial f}{\partial y} = x^2 + 6xy \quad \text{(treat x as constant)}$$

### 1.3 Gradient Vector

The **gradient** $\nabla f$ is a vector of all partial derivatives:

$$\nabla f = \left[\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \ldots, \frac{\partial f}{\partial x_n}\right]$$

**Geometric interpretation:** The gradient points in the direction of **steepest ascent**.

**For loss minimization:** Move in direction of $-\nabla L$ (negative gradient).

**Example:** For $f(x, y) = x^2 + y^2$:

$$\nabla f = [2x, 2y]$$

At point $(3, 4)$: $\nabla f(3, 4) = [6, 8]$ points away from origin (steepest ascent).

### 1.4 Hessian Matrix (Second Derivatives)

The **Hessian** $H$ contains all second partial derivatives:

$$H = \begin{bmatrix}
\frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} & \cdots \\
\frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} & \cdots \\
\vdots & \vdots & \ddots
\end{bmatrix}$$

**Use:** Determines concavity in multiple dimensions.
- If $H$ is **positive definite** at a critical point → local minimum
- If $H$ is **negative definite** → local maximum
- Mixed eigenvalues → saddle point

---

## 2. Gradient Descent: The Optimization Workhorse

### 2.1 Algorithm Review

**Goal:** Minimize $L(\theta)$ where $\theta = (\theta_1, \ldots, \theta_n)$ are parameters.

**Update rule:**

$$\theta_j := \theta_j - \alpha \frac{\partial L}{\partial \theta_j} \quad \text{for all } j$$

**Vectorized form:**

$$\theta := \theta - \alpha \nabla L(\theta)$$

where $\alpha$ is the learning rate.

### 2.2 Gradient Descent Variants

**Batch Gradient Descent:**
- Compute gradient using **all** training examples
- Update rule: $\theta := \theta - \alpha \frac{1}{m} \sum_{i=1}^{m} \nabla L^{(i)}(\theta)$
- **Pros:** Converges to global minimum for convex functions
- **Cons:** Slow for large datasets

**Stochastic Gradient Descent (SGD):**
- Compute gradient using **one** training example at a time
- Update rule: $\theta := \theta - \alpha \nabla L^{(i)}(\theta)$ (for random $i$)
- **Pros:** Fast, can escape local minima
- **Cons:** Noisy convergence

**Mini-Batch Gradient Descent:**
- Compute gradient using **small batch** of examples (e.g., 32, 64)
- Balance between batch and SGD
- **Standard in deep learning**

### 2.3 Learning Rate Schedules

Fixed learning rate $\alpha$ may be suboptimal:

**Time-based decay:**
$$\alpha_t = \frac{\alpha_0}{1 + dt}$$

**Exponential decay:**
$$\alpha_t = \alpha_0 e^{-kt}$$

**Adaptive methods (Adam, RMSProp):**
- Adjust learning rate per parameter based on gradient history

### 2.4 Momentum

Standard gradient descent can oscillate in narrow valleys.

**Momentum method:**
$$v_t = \beta v_{t-1} + (1-\beta) \nabla L(\theta_t)$$
$$\theta_{t+1} = \theta_t - \alpha v_t$$

where $\beta \in (0, 1)$ (typically 0.9) controls "inertia".

**Effect:** Smooths updates, accelerates convergence.

---

## 3. Taylor Series: Function Approximation

### 3.1 Taylor Series Review (Single Variable)

For a function $f(x)$ infinitely differentiable at $x = a$:

$$f(x) = f(a) + f'(a)(x-a) + \frac{f''(a)}{2!}(x-a)^2 + \frac{f'''(a)}{3!}(x-a)^3 + \cdots$$

**Maclaurin series** (special case $a = 0$):

$$f(x) = f(0) + f'(0)x + \frac{f''(0)}{2!}x^2 + \frac{f'''(0)}{3!}x^3 + \cdots$$

### 3.2 Common Taylor Series

**Exponential:**
$$e^x = 1 + x + \frac{x^2}{2!} + \frac{x^3}{3!} + \cdots = \sum_{n=0}^{\infty} \frac{x^n}{n!}$$

**Sine:**
$$\sin x = x - \frac{x^3}{3!} + \frac{x^5}{5!} - \frac{x^7}{7!} + \cdots = \sum_{n=0}^{\infty} \frac{(-1)^n x^{2n+1}}{(2n+1)!}$$

**Natural Logarithm:**
$$\ln(1 + x) = x - \frac{x^2}{2} + \frac{x^3}{3} - \frac{x^4}{4} + \cdots \quad (|x| < 1)$$

### 3.3 Applications in Machine Learning

**Sigmoid Activation:**
$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

For small $x$:
$$\sigma(x) \approx \frac{1}{2} + \frac{x}{4} - \frac{x^3}{48} + \cdots$$

**ReLU Approximation:**
$$\text{ReLU}(x) = \max(0, x)$$

Smooth approximation (softplus):
$$\text{softplus}(x) = \ln(1 + e^x) \approx x - \frac{e^{-x}}{2} \quad \text{(for large x)}$$

**Loss Function Approximation:**

Second-order Taylor expansion around current parameters $\theta^{(t)}$:

$$L(\theta) \approx L(\theta^{(t)}) + \nabla L(\theta^{(t)})^T (\theta - \theta^{(t)}) + \frac{1}{2}(\theta - \theta^{(t)})^T H(\theta^{(t)}) (\theta - \theta^{(t)})$$

This leads to **Newton's method** for optimization.

---

## 4. Newton's Method for Optimization

### 4.1 Newton's Method (Single Variable)

To find root of $f(x) = 0$:

$$x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)}$$

**For optimization:** To minimize $L(x)$, find where $L'(x) = 0$:

$$x_{n+1} = x_n - \frac{L'(x_n)}{L''(x_n)}$$

### 4.2 Newton's Method (Multivariable)

To minimize $L(\theta)$:

$$\theta_{n+1} = \theta_n - H^{-1}(\theta_n) \nabla L(\theta_n)$$

where $H$ is the Hessian matrix.

**Comparison with Gradient Descent:**
- **GD:** $\theta_{n+1} = \theta_n - \alpha \nabla L(\theta_n)$ (first-order)
- **Newton:** $\theta_{n+1} = \theta_n - H^{-1} \nabla L(\theta_n)$ (second-order)

**Pros:** Faster convergence (quadratic near minimum)
**Cons:** Computing and inverting Hessian is expensive ($O(n^3)$ for $n$ parameters)

### 4.3 Quasi-Newton Methods (BFGS, L-BFGS)

Approximate the Hessian using gradient information:
- **BFGS:** Builds approximation of $H^{-1}$ over iterations
- **L-BFGS:** Limited-memory version for large-scale problems

**Use case:** Standard optimizer for logistic regression, some neural networks.

---

## 5. Differential Equations Basics

### 5.1 What is a Differential Equation?

An equation involving a function and its derivatives:

$$\frac{dy}{dx} = f(x, y)$$

**Goal:** Find function $y(x)$ satisfying the equation.

### 5.2 Exponential Growth/Decay

**Model:** Rate of change proportional to current value:

$$\frac{dy}{dt} = ky$$

**Solution:**

$$y(t) = y_0 e^{kt}$$

where $y_0 = y(0)$ is the initial value.

**Applications:**
- **Population growth** (k > 0)
- **Radioactive decay** (k < 0)
- **Compound interest**
- **Loss convergence** in gradient descent

### 5.3 Logistic Growth

When growth has a carrying capacity $L$:

$$\frac{dy}{dt} = ky\left(1 - \frac{y}{L}\right)$$

**Solution:**

$$y(t) = \frac{L}{1 + Ae^{-kt}}$$

where $A$ depends on initial conditions.

**Applications:**
- **Sigmoid activation function** (neural networks)
- **Learning curves** (model performance vs training)
- **Epidemic models** (disease spread)

### 5.4 Gradient Flow

Gradient descent can be viewed as discrete approximation of:

$$\frac{d\theta}{dt} = -\nabla L(\theta)$$

**Continuous gradient flow:** Parameters evolve continuously to minimize loss.

**Connection to physics:** Ball rolling down a hill (potential energy = loss function).

---

## 6. Comprehensive Example: Linear Regression from Scratch

### 6.1 Problem Setup

**Dataset:** $\{(x^{(i)}, y^{(i)})\}_{i=1}^{m}$ where $x^{(i)} \in \mathbb{R}^n$, $y^{(i)} \in \mathbb{R}$

**Model (Hypothesis):**

$$h_\theta(x) = \theta_0 + \theta_1 x_1 + \cdots + \theta_n x_n = \theta^T x$$

**Loss Function (Mean Squared Error):**

$$L(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2$$

### 6.2 Gradient Computation

Partial derivatives:

$$\frac{\partial L}{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) x_j^{(i)}$$

**Vectorized gradient:**

$$\nabla L(\theta) = \frac{1}{m} X^T (X\theta - y)$$

where $X$ is the design matrix (rows are training examples).

### 6.3 Gradient Descent Implementation

**Algorithm:**
1. Initialize $\theta$ (e.g., zeros or small random values)
2. For each iteration:
   - Compute predictions: $\hat{y} = X\theta$
   - Compute gradient: $\nabla L = \frac{1}{m} X^T (\hat{y} - y)$
   - Update parameters: $\theta := \theta - \alpha \nabla L$
3. Repeat until convergence

**Convergence criteria:**
- $|L(\theta^{(t)}) - L(\theta^{(t-1)})| < \epsilon$ (loss change)
- $\|\nabla L(\theta^{(t)})\| < \epsilon$ (gradient magnitude)
- Maximum iterations reached

### 6.4 Feature Scaling

**Problem:** Features with different scales cause slow convergence.

**Solution:** Normalize features to similar ranges.

**Min-max scaling:**
$$x_{\text{scaled}} = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$$

**Standardization (z-score):**
$$x_{\text{scaled}} = \frac{x - \mu}{\sigma}$$

where $\mu$ is mean, $\sigma$ is standard deviation.

**Why it helps:** Gradient descent converges faster when loss function is more circular (isotropic).

---

## 7. End-to-End Application: Logistic Regression

### 7.1 Binary Classification Setup

**Goal:** Predict $y \in \{0, 1\}$ from features $x$.

**Model:**

$$h_\theta(x) = \sigma(\theta^T x) = \frac{1}{1 + e^{-\theta^T x}}$$

where $\sigma$ is the **sigmoid function** (Week 10).

**Properties:**
- $\sigma(z) \in (0, 1)$ (interpreted as probability)
- $\sigma'(z) = \sigma(z)(1 - \sigma(z))$ (convenient derivative!)

### 7.2 Loss Function (Cross-Entropy)

**Log-loss (Binary Cross-Entropy):**

$$L(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(h_\theta(x^{(i)})) + (1 - y^{(i)}) \log(1 - h_\theta(x^{(i)})) \right]$$

**Why this loss?**
- Derived from **maximum likelihood estimation**
- Penalizes confident wrong predictions heavily
- Convex for logistic regression (single global minimum)

### 7.3 Gradient Computation

**Key insight:** Despite different loss, gradient has same form as linear regression!

$$\frac{\partial L}{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) x_j^{(i)}$$

**Vectorized:**

$$\nabla L(\theta) = \frac{1}{m} X^T (\sigma(X\theta) - y)$$

### 7.4 Regularization

To prevent overfitting, add penalty for large weights:

**L2 regularization (Ridge):**

$$L_{\text{reg}}(\theta) = L(\theta) + \frac{\lambda}{2m} \sum_{j=1}^{n} \theta_j^2$$

**Gradient update:**

$$\frac{\partial L_{\text{reg}}}{\partial \theta_j} = \frac{\partial L}{\partial \theta_j} + \frac{\lambda}{m} \theta_j$$

**Effect:** Shrinks weights toward zero, reduces model complexity.

---

## 8. Neural Networks: Connecting All Concepts

### 8.1 Neural Network Architecture

**Single hidden layer:**

$$a^{[1]} = \sigma(W^{[1]} x + b^{[1]}) \quad \text{(hidden layer)}$$
$$\hat{y} = \sigma(W^{[2]} a^{[1]} + b^{[2]}) \quad \text{(output)}$$

where:
- $W^{[1]}, W^{[2]}$ are weight matrices
- $b^{[1]}, b^{[2]}$ are bias vectors
- $\sigma$ is activation function (sigmoid, ReLU, tanh)

### 8.2 Backpropagation: Chain Rule in Action

**Forward pass:** Compute predictions layer by layer.

**Backward pass:** Compute gradients using **chain rule** (Week 10):

$$\frac{\partial L}{\partial W^{[1]}} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial a^{[1]}} \cdot \frac{\partial a^{[1]}}{\partial W^{[1]}}$$

**Key formulas:**

For output layer:
$$\delta^{[2]} = (\hat{y} - y) \odot \sigma'(z^{[2]})$$

For hidden layer (backpropagating error):
$$\delta^{[1]} = (W^{[2]})^T \delta^{[2]} \odot \sigma'(z^{[1]})$$

Gradients:
$$\frac{\partial L}{\partial W^{[2]}} = \delta^{[2]} (a^{[1]})^T, \quad \frac{\partial L}{\partial W^{[1]}} = \delta^{[1]} x^T$$

### 8.3 Activation Functions Revisited

| Activation | Formula | Derivative | Use Case |
|-----------|---------|-----------|----------|
| Sigmoid | $\frac{1}{1+e^{-x}}$ | $\sigma(x)(1-\sigma(x))$ | Output layer (binary) |
| Tanh | $\frac{e^x - e^{-x}}{e^x + e^{-x}}$ | $1 - \tanh^2(x)$ | Hidden layers (zero-centered) |
| ReLU | $\max(0, x)$ | $1$ if $x>0$ else $0$ | Hidden layers (fast, simple) |
| Leaky ReLU | $\max(0.01x, x)$ | $1$ if $x>0$ else $0.01$ | Prevents "dead neurons" |

**Why derivatives matter:** Backpropagation requires $\sigma'(x)$ at every layer.

---

## 9. Putting It All Together: ML Pipeline

### 9.1 Complete Workflow

1. **Data Preparation**
   - Load and explore data
   - Handle missing values
   - Feature engineering (polynomials from Week 4, log transforms from Week 6)
   - Split train/validation/test sets

2. **Feature Scaling**
   - Standardization: $x_{\text{scaled}} = \frac{x - \mu}{\sigma}$
   - Min-max: $x_{\text{scaled}} = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$

3. **Model Initialization**
   - Choose architecture (linear, logistic, neural network)
   - Initialize parameters (small random values)
   - Set hyperparameters ($\alpha$, $\lambda$, batch size)

4. **Training Loop**
   ```
   for epoch in range(num_epochs):
       for batch in mini_batches:
           # Forward pass
           predictions = model(batch_X)
           loss = compute_loss(predictions, batch_y)

           # Backward pass (gradients)
           gradients = compute_gradients(loss)

           # Update parameters
           parameters -= learning_rate * gradients
   ```

5. **Evaluation**
   - Compute metrics on validation set (accuracy, F1, AUC)
   - Visualize learning curves (loss vs epochs)
   - Check for overfitting (train vs validation loss)

6. **Hyperparameter Tuning**
   - Grid search or random search
   - Adjust $\alpha$, $\lambda$, network architecture
   - Cross-validation for robust estimates

7. **Testing**
   - Evaluate on held-out test set (once!)
   - Report final metrics

### 9.2 Connecting to Course Concepts

| Week | Concept | ML Application |
|------|---------|----------------|
| 4 | Polynomials | Polynomial regression, feature engineering |
| 5 | Functions | Model as function approximation |
| 6 | Exponential/log | Sigmoid, softmax, log-loss |
| 7 | Trigonometry | Fourier features, cosine similarity |
| 8 | Series | Taylor approximations, weight initialization |
| 9 | Limits/Continuity | Loss surface analysis, gradient existence |
| 10 | Derivatives | Gradient descent, backpropagation |
| 11 | Integration | Expected values, normalizing constants |

---

## 10. Summary and Looking Forward

### 10.1 Core Mathematical Toolkit for Data Science

**Functions and Transformations:**
- Understanding domain, range, composition
- Exponential, logarithmic, trigonometric functions
- Activation functions as nonlinear transformations

**Calculus:**
- **Derivatives:** Measure rates of change, optimize parameters
- **Partial derivatives:** Handle multivariable functions (loss surfaces)
- **Gradients:** Point in direction of steepest ascent/descent
- **Integration:** Compute probabilities, expected values

**Series and Approximations:**
- **Taylor series:** Approximate complex functions
- **Convergence:** Ensure algorithms terminate
- **Sequences:** Iterative optimization (gradient descent)

**Optimization:**
- **Gradient descent:** First-order optimization (workhorse of ML)
- **Newton's method:** Second-order (faster but expensive)
- **Stochastic variants:** Scale to large datasets

### 10.2 Next Steps in Your Learning Journey

**Immediate Next Courses (Foundation Level):**
- **Statistics I & II:** Probability distributions, hypothesis testing, confidence intervals
- **Python Programming:** Implement ML algorithms from scratch
- **Computational Thinking:** Algorithm design, complexity analysis

**Diploma Level (After Foundation):**
- **Linear Algebra:** Matrices, eigenvectors, SVD (critical for ML)
- **Probability Theory:** Bayesian inference, Markov chains
- **Machine Learning:** Supervised/unsupervised learning, model evaluation

**BSc/BS Level:**
- **Advanced ML:** Deep learning, reinforcement learning
- **Optimization Theory:** Convex optimization, duality
- **Statistical Learning Theory:** PAC learning, VC dimension

### 10.3 Recommended Practice

**Implement from Scratch:**
- Linear regression with gradient descent
- Logistic regression with regularization
- 2-layer neural network with backpropagation

**Explore Real Datasets:**
- Kaggle competitions (Titanic, House Prices)
- UCI Machine Learning Repository
- scikit-learn built-in datasets

**Visualize Concepts:**
- Plot loss surfaces (3D plots)
- Animate gradient descent convergence
- Visualize decision boundaries

**Read Further:**
- *Pattern Recognition and Machine Learning* by Bishop
- *Deep Learning* by Goodfellow, Bengio, Courville
- *Mathematics for Machine Learning* by Deisenroth et al.

---

## 11. Final Worked Example: Ridge Regression

**Problem:** Predict house prices from features (area, bedrooms, age).

**Dataset:** $m = 100$ houses, $n = 3$ features

**Model:**

$$h_\theta(x) = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \theta_3 x_3$$

**Loss (with L2 regularization):**

$$L(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2 + \frac{\lambda}{2m} \sum_{j=1}^{3} \theta_j^2$$

**Gradient:**

$$\frac{\partial L}{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) x_j^{(i)} + \frac{\lambda}{m} \theta_j \quad (j \geq 1)$$

$$\frac{\partial L}{\partial \theta_0} = \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) \quad \text{(no regularization on bias)}$$

**Algorithm (Gradient Descent with Momentum):**

```
Initialize: θ = 0, v = 0
For t = 1 to max_iterations:
    # Compute gradient
    g = (1/m) X^T (Xθ - y) + (λ/m) θ

    # Momentum update
    v = β*v + (1-β)*g

    # Parameter update
    θ = θ - α*v

    # Check convergence
    If ||g|| < ε: break
```

**Hyperparameters:**
- Learning rate: $\alpha = 0.01$
- Regularization: $\lambda = 0.1$
- Momentum: $\beta = 0.9$

**Expected outcome:**
- Converges in ~500 iterations
- Reduced overfitting compared to unregularized regression
- Test set RMSE improves by ~15%

---

## 12. Conclusion: Mathematics Powers Machine Learning

**Key Takeaway:** Every ML algorithm relies on the mathematics you've learned in Weeks 4-11.

**From Theory to Practice:**
- **Functions** → Models that map inputs to outputs
- **Derivatives** → Gradients that guide optimization
- **Integration** → Probabilities and expected values
- **Series** → Function approximations and iterative algorithms

**Your Mathematical Foundation:**

You now understand:
- How gradient descent minimizes loss functions (calculus)
- Why activation functions need specific properties (continuity, derivatives)
- How Taylor series approximate complex functions
- Why feature scaling improves convergence (loss surface geometry)
- How probability distributions connect to integrals

**Congratulations on completing BSMA1001 Mathematics I!**

You've built a solid mathematical foundation for data science. As you progress through Statistics, Linear Algebra, and Machine Learning courses, you'll see these concepts applied repeatedly. The calculus, functions, and series you mastered here are not just abstract theory—they're the **engines of modern AI**.

**Keep practicing, keep learning, and enjoy the journey!** 🚀

---

## References

1. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
2. Murphy, K. P. (2022). *Probabilistic Machine Learning: An Introduction*. MIT Press.
3. Deisenroth, M. P., Faisal, A. A., & Ong, C. S. (2020). *Mathematics for Machine Learning*. Cambridge University Press.
4. Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*. Cambridge University Press.
5. IIT Madras BS in Data Science - BSMA1001 Course Materials

---

**End of Week 12 Notes - End of BSMA1001 Mathematics I Course**
