---
Date: 2025-11-21
Course: BSMA1001 - Mathematics for Data Science I
Level: Foundation
Week: 7 of 12
Source: IIT Madras Mathematics for Data Science I Week 7
Topic Area: Mathematics
Tags: #BSMA1001 #Trigonometry #Week7 #Foundation
---

# Week 7: Trigonometric Functions

## Overview

Trigonometric functions are fundamental to mathematics, appearing in periodic phenomena, signal processing, Fourier analysis, and machine learning. While originally developed to solve problems in geometry and astronomy, these functions are now essential tools in data science for analyzing oscillatory patterns, working with angles and rotations, and decomposing complex signals.

**Learning Objectives:**
- Understand angle measurement in degrees and radians
- Master the six trigonometric functions and their relationships
- Work with trigonometric identities and formulas
- Solve trigonometric equations
- Apply trigonometry to periodic data and signal analysis
- Understand Fourier series foundations

**Key Concepts:** Radians, sine, cosine, tangent, unit circle, periodicity, trigonometric identities, inverse trigonometric functions

---

## 1. Angle Measurement

### 1.1 Degrees and Radians

**Degree Measure**: Complete circle = 360°

**Radian Measure**: Angle subtended by arc equal to radius

**Definition**: One radian is the angle when arc length equals radius.

**Full Circle**: $2\pi$ radians = 360°

**Conversion Formulas:**
$$\text{Radians} = \text{Degrees} \times \frac{\pi}{180}$$

$$\text{Degrees} = \text{Radians} \times \frac{180}{\pi}$$

**Example 1.1: Converting Between Degrees and Radians**

**a) Convert 60° to radians:**

$$60° = 60 \times \frac{\pi}{180} = \frac{\pi}{3} \text{ radians}$$

**b) Convert $\frac{3\pi}{4}$ radians to degrees:**

$$\frac{3\pi}{4} = \frac{3\pi}{4} \times \frac{180}{\pi} = 135°$$

### 1.2 Common Angle Conversions

| Degrees | Radians | Notable |
|---------|---------|---------|
| 0° | 0 | Zero angle |
| 30° | $\frac{\pi}{6}$ | |
| 45° | $\frac{\pi}{4}$ | |
| 60° | $\frac{\pi}{3}$ | |
| 90° | $\frac{\pi}{2}$ | Right angle |
| 120° | $\frac{2\pi}{3}$ | |
| 135° | $\frac{3\pi}{4}$ | |
| 150° | $\frac{5\pi}{6}$ | |
| 180° | $\pi$ | Straight angle |
| 270° | $\frac{3\pi}{2}$ | |
| 360° | $2\pi$ | Full circle |

**Why Radians?**
- Natural unit in calculus: $\frac{d}{dx}[\sin(x)] = \cos(x)$ (only true when $x$ is in radians)
- Arc length formula: $s = r\theta$ (simple when using radians)
- Used universally in mathematics and data science

---

## 2. The Unit Circle

### 2.1 Definition

The **unit circle** is a circle with radius 1 centered at the origin.

**Equation**: $x^2 + y^2 = 1$

**Angle $\theta$**: Measured counterclockwise from positive $x$-axis

**Point on Circle**: $P(\cos\theta, \sin\theta)$

### 2.2 Trigonometric Functions from Unit Circle

For angle $\theta$ in standard position:

$$\cos(\theta) = x\text{-coordinate of point on unit circle}$$
$$\sin(\theta) = y\text{-coordinate of point on unit circle}$$
$$\tan(\theta) = \frac{\sin(\theta)}{\cos(\theta)} = \frac{y}{x}$$

**Example 2.1: Unit Circle Values**

For $\theta = \frac{\pi}{4}$ (45°):

Point on unit circle: $\left(\frac{\sqrt{2}}{2}, \frac{\sqrt{2}}{2}\right)$

$$\cos\left(\frac{\pi}{4}\right) = \frac{\sqrt{2}}{2} \approx 0.707$$

$$\sin\left(\frac{\pi}{4}\right) = \frac{\sqrt{2}}{2} \approx 0.707$$

$$\tan\left(\frac{\pi}{4}\right) = \frac{\sin(\pi/4)}{\cos(\pi/4)} = 1$$

### 2.3 Special Angles Reference

**At $\theta = 0$ (0°):**
- $\cos(0) = 1$
- $\sin(0) = 0$
- $\tan(0) = 0$

**At $\theta = \frac{\pi}{6}$ (30°):**
- $\cos(\pi/6) = \frac{\sqrt{3}}{2}$
- $\sin(\pi/6) = \frac{1}{2}$
- $\tan(\pi/6) = \frac{1}{\sqrt{3}} = \frac{\sqrt{3}}{3}$

**At $\theta = \frac{\pi}{4}$ (45°):**
- $\cos(\pi/4) = \frac{\sqrt{2}}{2}$
- $\sin(\pi/4) = \frac{\sqrt{2}}{2}$
- $\tan(\pi/4) = 1$

**At $\theta = \frac{\pi}{3}$ (60°):**
- $\cos(\pi/3) = \frac{1}{2}$
- $\sin(\pi/3) = \frac{\sqrt{3}}{2}$
- $\tan(\pi/3) = \sqrt{3}$

**At $\theta = \frac{\pi}{2}$ (90°):**
- $\cos(\pi/2) = 0$
- $\sin(\pi/2) = 1$
- $\tan(\pi/2)$ = undefined

**Mnemonic for 0°, 30°, 45°, 60°, 90°:**

For sine: $\sin(\theta) = \frac{\sqrt{n}}{2}$ where $n = 0, 1, 2, 3, 4$

For cosine: reverse the pattern

---

## 3. Six Trigonometric Functions

### 3.1 Primary Functions

**Sine**: $\sin(\theta) = \frac{\text{opposite}}{\text{hypotenuse}}$ (in right triangle)

**Cosine**: $\cos(\theta) = \frac{\text{adjacent}}{\text{hypotenuse}}$

**Tangent**: $\tan(\theta) = \frac{\sin(\theta)}{\cos(\theta)} = \frac{\text{opposite}}{\text{adjacent}}$

### 3.2 Reciprocal Functions

**Cosecant**: $\csc(\theta) = \frac{1}{\sin(\theta)}$

**Secant**: $\sec(\theta) = \frac{1}{\cos(\theta)}$

**Cotangent**: $\cot(\theta) = \frac{1}{\tan(\theta)} = \frac{\cos(\theta)}{\sin(\theta)}$

### 3.3 Properties of Trigonometric Functions

| Function | Domain | Range | Period |
|----------|--------|-------|--------|
| $\sin(x)$ | $\mathbb{R}$ | $[-1, 1]$ | $2\pi$ |
| $\cos(x)$ | $\mathbb{R}$ | $[-1, 1]$ | $2\pi$ |
| $\tan(x)$ | $x \neq \frac{\pi}{2} + n\pi$ | $\mathbb{R}$ | $\pi$ |
| $\csc(x)$ | $x \neq n\pi$ | $(-\infty, -1] \cup [1, \infty)$ | $2\pi$ |
| $\sec(x)$ | $x \neq \frac{\pi}{2} + n\pi$ | $(-\infty, -1] \cup [1, \infty)$ | $2\pi$ |
| $\cot(x)$ | $x \neq n\pi$ | $\mathbb{R}$ | $\pi$ |

**Example 3.1: Evaluating Trigonometric Functions**

Find all six trigonometric functions of $\theta = \frac{2\pi}{3}$ (120°).

**Solution:**

Reference angle: $\pi - \frac{2\pi}{3} = \frac{\pi}{3}$

Quadrant II: sine positive, cosine and tangent negative

$$\sin\left(\frac{2\pi}{3}\right) = \sin\left(\frac{\pi}{3}\right) = \frac{\sqrt{3}}{2}$$

$$\cos\left(\frac{2\pi}{3}\right) = -\cos\left(\frac{\pi}{3}\right) = -\frac{1}{2}$$

$$\tan\left(\frac{2\pi}{3}\right) = \frac{\sin(2\pi/3)}{\cos(2\pi/3)} = \frac{\sqrt{3}/2}{-1/2} = -\sqrt{3}$$

$$\csc\left(\frac{2\pi}{3}\right) = \frac{1}{\sin(2\pi/3)} = \frac{2}{\sqrt{3}} = \frac{2\sqrt{3}}{3}$$

$$\sec\left(\frac{2\pi}{3}\right) = \frac{1}{\cos(2\pi/3)} = -2$$

$$\cot\left(\frac{2\pi}{3}\right) = \frac{1}{\tan(2\pi/3)} = -\frac{1}{\sqrt{3}} = -\frac{\sqrt{3}}{3}$$

---

## 4. Fundamental Trigonometric Identities

### 4.1 Pythagorean Identities

**Primary Identity:**
$$\sin^2(\theta) + \cos^2(\theta) = 1$$

**Derived Identities:**
$$1 + \tan^2(\theta) = \sec^2(\theta)$$

$$1 + \cot^2(\theta) = \csc^2(\theta)$$

**Proof of Primary Identity:**

From unit circle: point $(\cos\theta, \sin\theta)$ satisfies $x^2 + y^2 = 1$

Therefore: $\cos^2(\theta) + \sin^2(\theta) = 1$ ∎

### 4.2 Reciprocal Identities

$$\csc(\theta) = \frac{1}{\sin(\theta)}$$

$$\sec(\theta) = \frac{1}{\cos(\theta)}$$

$$\cot(\theta) = \frac{1}{\tan(\theta)}$$

### 4.3 Quotient Identities

$$\tan(\theta) = \frac{\sin(\theta)}{\cos(\theta)}$$

$$\cot(\theta) = \frac{\cos(\theta)}{\sin(\theta)}$$

### 4.4 Even-Odd Identities

**Even Functions** (symmetric about y-axis):
$$\cos(-\theta) = \cos(\theta)$$
$$\sec(-\theta) = \sec(\theta)$$

**Odd Functions** (symmetric about origin):
$$\sin(-\theta) = -\sin(\theta)$$
$$\tan(-\theta) = -\tan(\theta)$$
$$\csc(-\theta) = -\csc(\theta)$$
$$\cot(-\theta) = -\cot(\theta)$$

**Example 4.1: Using Pythagorean Identity**

If $\sin(\theta) = \frac{3}{5}$ and $\theta$ is in Quadrant II, find $\cos(\theta)$ and $\tan(\theta)$.

**Solution:**

Using $\sin^2(\theta) + \cos^2(\theta) = 1$:

$$\left(\frac{3}{5}\right)^2 + \cos^2(\theta) = 1$$

$$\frac{9}{25} + \cos^2(\theta) = 1$$

$$\cos^2(\theta) = 1 - \frac{9}{25} = \frac{16}{25}$$

$$\cos(\theta) = \pm\frac{4}{5}$$

Since $\theta$ is in Quadrant II, cosine is negative:
$$\cos(\theta) = -\frac{4}{5}$$

Then:
$$\tan(\theta) = \frac{\sin(\theta)}{\cos(\theta)} = \frac{3/5}{-4/5} = -\frac{3}{4}$$

---

## 5. Sum and Difference Formulas

### 5.1 Sine Sum and Difference

$$\sin(\alpha + \beta) = \sin(\alpha)\cos(\beta) + \cos(\alpha)\sin(\beta)$$

$$\sin(\alpha - \beta) = \sin(\alpha)\cos(\beta) - \cos(\alpha)\sin(\beta)$$

### 5.2 Cosine Sum and Difference

$$\cos(\alpha + \beta) = \cos(\alpha)\cos(\beta) - \sin(\alpha)\sin(\beta)$$

$$\cos(\alpha - \beta) = \cos(\alpha)\cos(\beta) + \sin(\alpha)\sin(\beta)$$

### 5.3 Tangent Sum and Difference

$$\tan(\alpha + \beta) = \frac{\tan(\alpha) + \tan(\beta)}{1 - \tan(\alpha)\tan(\beta)}$$

$$\tan(\alpha - \beta) = \frac{\tan(\alpha) - \tan(\beta)}{1 + \tan(\alpha)\tan(\beta)}$$

**Example 5.1: Using Sum Formula**

Find the exact value of $\sin(75°)$.

**Solution:**

Express as sum of special angles: $75° = 45° + 30°$

$$\sin(75°) = \sin(45° + 30°)$$

Using sum formula:
\begin{align*}
\sin(75°) &= \sin(45°)\cos(30°) + \cos(45°)\sin(30°) \\
&= \frac{\sqrt{2}}{2} \cdot \frac{\sqrt{3}}{2} + \frac{\sqrt{2}}{2} \cdot \frac{1}{2} \\
&= \frac{\sqrt{6}}{4} + \frac{\sqrt{2}}{4} \\
&= \frac{\sqrt{6} + \sqrt{2}}{4}
\end{align*}

---

## 6. Double Angle and Half Angle Formulas

### 6.1 Double Angle Formulas

**Sine:**
$$\sin(2\theta) = 2\sin(\theta)\cos(\theta)$$

**Cosine (three forms):**
$$\cos(2\theta) = \cos^2(\theta) - \sin^2(\theta)$$
$$\cos(2\theta) = 2\cos^2(\theta) - 1$$
$$\cos(2\theta) = 1 - 2\sin^2(\theta)$$

**Tangent:**
$$\tan(2\theta) = \frac{2\tan(\theta)}{1 - \tan^2(\theta)}$$

### 6.2 Half Angle Formulas

$$\sin\left(\frac{\theta}{2}\right) = \pm\sqrt{\frac{1 - \cos(\theta)}{2}}$$

$$\cos\left(\frac{\theta}{2}\right) = \pm\sqrt{\frac{1 + \cos(\theta)}{2}}$$

$$\tan\left(\frac{\theta}{2}\right) = \frac{\sin(\theta)}{1 + \cos(\theta)} = \frac{1 - \cos(\theta)}{\sin(\theta)}$$

(Sign depends on quadrant of $\frac{\theta}{2}$)

**Example 6.1: Double Angle Formula**

If $\cos(\theta) = \frac{3}{5}$ and $0 < \theta < \frac{\pi}{2}$, find $\sin(2\theta)$ and $\cos(2\theta)$.

**Solution:**

First find $\sin(\theta)$ using Pythagorean identity:
$$\sin^2(\theta) = 1 - \cos^2(\theta) = 1 - \frac{9}{25} = \frac{16}{25}$$

Since $\theta$ is in Quadrant I: $\sin(\theta) = \frac{4}{5}$

**For $\sin(2\theta)$:**
$$\sin(2\theta) = 2\sin(\theta)\cos(\theta) = 2 \cdot \frac{4}{5} \cdot \frac{3}{5} = \frac{24}{25}$$

**For $\cos(2\theta)$:**
$$\cos(2\theta) = \cos^2(\theta) - \sin^2(\theta) = \frac{9}{25} - \frac{16}{25} = -\frac{7}{25}$$

---

## 7. Product-to-Sum and Sum-to-Product Formulas

### 7.1 Product-to-Sum

$$\sin(\alpha)\cos(\beta) = \frac{1}{2}[\sin(\alpha+\beta) + \sin(\alpha-\beta)]$$

$$\cos(\alpha)\sin(\beta) = \frac{1}{2}[\sin(\alpha+\beta) - \sin(\alpha-\beta)]$$

$$\cos(\alpha)\cos(\beta) = \frac{1}{2}[\cos(\alpha-\beta) + \cos(\alpha+\beta)]$$

$$\sin(\alpha)\sin(\beta) = \frac{1}{2}[\cos(\alpha-\beta) - \cos(\alpha+\beta)]$$

### 7.2 Sum-to-Product

$$\sin(\alpha) + \sin(\beta) = 2\sin\left(\frac{\alpha+\beta}{2}\right)\cos\left(\frac{\alpha-\beta}{2}\right)$$

$$\sin(\alpha) - \sin(\beta) = 2\cos\left(\frac{\alpha+\beta}{2}\right)\sin\left(\frac{\alpha-\beta}{2}\right)$$

$$\cos(\alpha) + \cos(\beta) = 2\cos\left(\frac{\alpha+\beta}{2}\right)\cos\left(\frac{\alpha-\beta}{2}\right)$$

$$\cos(\alpha) - \cos(\beta) = -2\sin\left(\frac{\alpha+\beta}{2}\right)\sin\left(\frac{\alpha-\beta}{2}\right)$$

**Purpose**: These formulas are essential in Fourier analysis and signal processing.

---

## 8. Inverse Trigonometric Functions

### 8.1 Definitions

**Arcsine**: $y = \arcsin(x) = \sin^{-1}(x)$ means $\sin(y) = x$
- Domain: $[-1, 1]$
- Range: $\left[-\frac{\pi}{2}, \frac{\pi}{2}\right]$

**Arccosine**: $y = \arccos(x) = \cos^{-1}(x)$ means $\cos(y) = x$
- Domain: $[-1, 1]$
- Range: $[0, \pi]$

**Arctangent**: $y = \arctan(x) = \tan^{-1}(x)$ means $\tan(y) = x$
- Domain: $\mathbb{R}$
- Range: $\left(-\frac{\pi}{2}, \frac{\pi}{2}\right)$

**Note**: $\sin^{-1}(x)$ does NOT mean $\frac{1}{\sin(x)}$ (that's $\csc(x)$)

### 8.2 Properties

**Inverse Relationships:**
$$\sin(\arcsin(x)) = x \text{ for } x \in [-1,1]$$
$$\arcsin(\sin(x)) = x \text{ for } x \in \left[-\frac{\pi}{2}, \frac{\pi}{2}\right]$$

(Similar for cos and tan)

**Example 8.1: Evaluating Inverse Trig Functions**

**a)** $\arcsin\left(\frac{1}{2}\right) = \frac{\pi}{6}$ (because $\sin(\pi/6) = 1/2$ and $\pi/6$ is in range)

**b)** $\arccos\left(-\frac{\sqrt{2}}{2}\right) = \frac{3\pi}{4}$ (because $\cos(3\pi/4) = -\sqrt{2}/2$ and $3\pi/4 \in [0,\pi]$)

**c)** $\arctan(1) = \frac{\pi}{4}$ (because $\tan(\pi/4) = 1$)

**d)** $\sin(\arccos(3/5))$

**Solution**: Let $\theta = \arccos(3/5)$, so $\cos(\theta) = 3/5$

Using Pythagorean identity: $\sin(\theta) = \sqrt{1 - (3/5)^2} = \sqrt{16/25} = 4/5$

Therefore: $\sin(\arccos(3/5)) = 4/5$

---

## 9. Solving Trigonometric Equations

### 9.1 Basic Strategies

1. **Isolate the trigonometric function**
2. **Find reference angle**
3. **Determine all solutions in given interval**
4. **Use periodicity for general solution**

**Example 9.1: Basic Equation**

Solve $2\sin(x) - 1 = 0$ for $x \in [0, 2\pi)$.

**Solution:**

$$2\sin(x) = 1$$
$$\sin(x) = \frac{1}{2}$$

Reference angle: $\sin^{-1}(1/2) = \frac{\pi}{6}$

Sine is positive in Quadrants I and II:
$$x = \frac{\pi}{6} \text{ or } x = \pi - \frac{\pi}{6} = \frac{5\pi}{6}$$

**Answer**: $x = \frac{\pi}{6}, \frac{5\pi}{6}$

### 9.2 Using Identities

**Example 9.2: Using Double Angle**

Solve $\cos(2x) = \cos(x)$ for $x \in [0, 2\pi)$.

**Solution:**

Using $\cos(2x) = 2\cos^2(x) - 1$:
$$2\cos^2(x) - 1 = \cos(x)$$
$$2\cos^2(x) - \cos(x) - 1 = 0$$

Let $u = \cos(x)$:
$$2u^2 - u - 1 = 0$$
$$(2u + 1)(u - 1) = 0$$

So $u = -\frac{1}{2}$ or $u = 1$

**Case 1**: $\cos(x) = 1 \implies x = 0$

**Case 2**: $\cos(x) = -\frac{1}{2} \implies x = \frac{2\pi}{3}, \frac{4\pi}{3}$

**Answer**: $x = 0, \frac{2\pi}{3}, \frac{4\pi}{3}$

### 9.3 Quadratic Form

**Example 9.3: Quadratic in Sine**

Solve $2\sin^2(x) - 3\sin(x) + 1 = 0$ for $x \in [0, 2\pi)$.

**Solution:**

Factor: $(2\sin(x) - 1)(\sin(x) - 1) = 0$

**Case 1**: $\sin(x) = \frac{1}{2} \implies x = \frac{\pi}{6}, \frac{5\pi}{6}$

**Case 2**: $\sin(x) = 1 \implies x = \frac{\pi}{2}$

**Answer**: $x = \frac{\pi}{6}, \frac{\pi}{2}, \frac{5\pi}{6}$

---

## 10. Applications in Data Science

### 10.1 Periodic Phenomena

**Modeling Cyclical Data:**

General form: $y = A\sin(B(x - C)) + D$ or $y = A\cos(B(x - C)) + D$

Where:
- $A$: **Amplitude** (half distance from max to min)
- $B$: Frequency ($\text{Period} = \frac{2\pi}{B}$)
- $C$: **Phase shift** (horizontal shift)
- $D$: **Vertical shift** (midline)

**Example 10.1: Temperature Modeling**

Daily temperature can be modeled as:
$$T(t) = 15\cos\left(\frac{\pi}{12}(t - 15)\right) + 25$$

Where $t$ is hours after midnight.

**Interpretation:**
- Average temperature: 25°C
- Temperature variation: ±15°C
- Period: $\frac{2\pi}{\pi/12} = 24$ hours
- Highest temp at $t = 15$ (3 PM)

### 10.2 Fourier Analysis

**Fourier Series**: Any periodic function can be decomposed into sum of sines and cosines:

$$f(x) = a_0 + \sum_{n=1}^{\infty}\left[a_n\cos(nx) + b_n\sin(nx)\right]$$

**Applications:**
- **Signal processing**: Audio, images
- **Time series decomposition**: Seasonal patterns
- **Data compression**: JPEG, MP3
- **Filtering**: Remove noise

**Key Insight**: Complex periodic patterns = sum of simple sine/cosine waves

### 10.3 Wave Phenomena

**Combining Waves:**

If $y_1 = A\sin(\omega t)$ and $y_2 = B\sin(\omega t + \phi)$

Then: $y = y_1 + y_2$ (superposition principle)

**Applications:**
- Sound waves (interference, beats)
- Electromagnetic waves
- Quantum mechanics (wave functions)
- Image processing (frequency domain)

### 10.4 Distance and Angles

**Cosine Similarity** in machine learning:

$$\text{similarity} = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \|\mathbf{B}\|} = \cos(\theta)$$

Where $\theta$ is angle between vectors $\mathbf{A}$ and $\mathbf{B}$.

**Used in:**
- Document similarity
- Recommendation systems
- Image recognition
- Natural language processing

**Range**: $[-1, 1]$
- $\cos(\theta) = 1$: Vectors point same direction (identical)
- $\cos(\theta) = 0$: Vectors perpendicular (unrelated)
- $\cos(\theta) = -1$: Vectors point opposite directions

### 10.5 Rotation Matrices

**2D Rotation** by angle $\theta$:

$$\begin{bmatrix} x' \\ y' \end{bmatrix} = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix}$$

**Applications:**
- Computer graphics
- Image transformations
- Robot kinematics
- Principal Component Analysis (PCA)

### 10.6 Circular Statistics

**Mean Direction** of angles $\theta_1, \ldots, \theta_n$:

$$\bar{\theta} = \arctan\left(\frac{\sum\sin\theta_i}{\sum\cos\theta_i}\right)$$

**Used for:**
- Wind direction analysis
- Animal migration patterns
- Molecular biology (dihedral angles)

---

## 11. Graphs of Trigonometric Functions

### 11.1 Sine and Cosine

**$y = \sin(x)$:**
- Starts at $(0, 0)$
- Maximum at $x = \frac{\pi}{2}$
- Period: $2\pi$
- Amplitude: 1
- Midline: $y = 0$

**$y = \cos(x)$:**
- Starts at $(0, 1)$
- Maximum at $x = 0$
- Period: $2\pi$
- Amplitude: 1
- Midline: $y = 0$

**Relationship**: $\cos(x) = \sin\left(x + \frac{\pi}{2}\right)$ (cosine is sine shifted left by $\pi/2$)

### 11.2 Tangent and Cotangent

**$y = \tan(x)$:**
- Period: $\pi$
- Vertical asymptotes at $x = \pm\frac{\pi}{2}, \pm\frac{3\pi}{2}, \ldots$
- Passes through origin
- Range: $\mathbb{R}$

**$y = \cot(x)$:**
- Period: $\pi$
- Vertical asymptotes at $x = 0, \pm\pi, \pm 2\pi, \ldots$
- Range: $\mathbb{R}$

### 11.3 Transformations

**Amplitude**: $y = A\sin(x)$ stretches vertically by $|A|$
- If $A < 0$, also reflects across x-axis

**Period**: $y = \sin(Bx)$ has period $\frac{2\pi}{|B|}$
- Larger $|B|$ → shorter period (faster oscillation)

**Phase Shift**: $y = \sin(x - C)$ shifts right by $C$

**Vertical Shift**: $y = \sin(x) + D$ shifts up by $D$

---

## 12. Common Pitfalls and Misconceptions

### 12.1 Radian vs Degree Mode

❌ **Wrong**: Using degrees when formula requires radians
✅ **Correct**: Always use radians in calculus and most data science applications

**Example**: $\sin(30)$ in radians ≠ $\sin(30°)$
- $\sin(30 \text{ radians}) \approx -0.988$
- $\sin(30°) = 0.5$

### 12.2 Domain Restrictions

❌ **Wrong**: $\tan(\pi/2) = \text{some finite number}$
✅ **Correct**: $\tan(\pi/2)$ is **undefined** (vertical asymptote)

### 12.3 Inverse Function Notation

❌ **Wrong**: $\sin^{-1}(x) = \frac{1}{\sin(x)}$
✅ **Correct**: $\sin^{-1}(x) = \arcsin(x)$ (inverse function, not reciprocal)

Reciprocal is $\csc(x) = \frac{1}{\sin(x)}$

### 12.4 All Solutions vs Principal Value

❌ **Wrong**: $\sin(x) = \frac{1}{2}$ has only one solution $x = \frac{\pi}{6}$
✅ **Correct**: Infinitely many solutions: $x = \frac{\pi}{6} + 2\pi n$ or $x = \frac{5\pi}{6} + 2\pi n$ for integer $n$

Always specify interval when solving equations!

### 12.5 Addition Formulas

❌ **Wrong**: $\sin(\alpha + \beta) = \sin(\alpha) + \sin(\beta)$
✅ **Correct**: $\sin(\alpha + \beta) = \sin(\alpha)\cos(\beta) + \cos(\alpha)\sin(\beta)$

This is NOT distributive!

### 12.6 Pythagorean Identity

❌ **Wrong**: $\sin^2(x) + \cos^2(x) = 2$
✅ **Correct**: $\sin^2(x) + \cos^2(x) = 1$ always

---

## 13. Worked Examples

### Example 13.1: Comprehensive Problem

**Problem**: Prove that $\frac{1 - \cos(2x)}{\sin(2x)} = \tan(x)$

**Solution**:

Starting with left side:
$$\frac{1 - \cos(2x)}{\sin(2x)}$$

Use double angle formulas:
- $\cos(2x) = 1 - 2\sin^2(x)$
- $\sin(2x) = 2\sin(x)\cos(x)$

Substitute:
$$\frac{1 - (1 - 2\sin^2(x))}{2\sin(x)\cos(x)} = \frac{2\sin^2(x)}{2\sin(x)\cos(x)}$$

Simplify:
$$\frac{2\sin^2(x)}{2\sin(x)\cos(x)} = \frac{\sin(x)}{\cos(x)} = \tan(x)$$ ∎

### Example 13.2: Application Problem

**Problem**: A Ferris wheel has radius 50 feet and completes one rotation every 60 seconds. If you board at the bottom (ground level), write a function for your height $h(t)$ above ground at time $t$ seconds.

**Solution**:

**Center height**: $50$ feet (radius above ground)

**Radius/Amplitude**: $50$ feet

**Period**: $60$ seconds, so $B = \frac{2\pi}{60} = \frac{\pi}{30}$

**Starting position**: Bottom, so use negative cosine (shifted)

**Model**:
$$h(t) = -50\cos\left(\frac{\pi}{30}t\right) + 50$$

Or equivalently:
$$h(t) = 50 - 50\cos\left(\frac{\pi}{30}t\right) = 50\left(1 - \cos\left(\frac{\pi}{30}t\right)\right)$$

**Verification**:
- At $t = 0$: $h(0) = 50(1 - 1) = 0$ feet ✓ (ground)
- At $t = 30$: $h(30) = 50(1 - \cos(\pi)) = 50(1 - (-1)) = 100$ feet ✓ (top)
- At $t = 60$: $h(60) = 50(1 - \cos(2\pi)) = 0$ feet ✓ (back to ground)

### Example 13.3: Proving an Identity

**Problem**: Prove $\tan(x) + \cot(x) = \frac{2}{\sin(2x)}$

**Solution**:

Left side:
$$\tan(x) + \cot(x) = \frac{\sin(x)}{\cos(x)} + \frac{\cos(x)}{\sin(x)}$$

Common denominator:
$$= \frac{\sin^2(x) + \cos^2(x)}{\sin(x)\cos(x)}$$

Use Pythagorean identity:
$$= \frac{1}{\sin(x)\cos(x)}$$

Multiply by $\frac{2}{2}$:
$$= \frac{2}{2\sin(x)\cos(x)}$$

Use double angle formula $\sin(2x) = 2\sin(x)\cos(x)$:
$$= \frac{2}{\sin(2x)}$$ ∎

---

## 14. Practice Problems

### Basic Level

**Problem 1**: Convert to radians: 135°, 210°, 315°

**Problem 2**: Convert to degrees: $\frac{5\pi}{6}$, $\frac{7\pi}{4}$, $\frac{11\pi}{6}$

**Problem 3**: Evaluate without calculator: $\sin(\pi/3)$, $\cos(3\pi/4)$, $\tan(\pi/6)$

**Problem 4**: If $\sin(\theta) = \frac{5}{13}$ and $\theta$ is in Quadrant I, find $\cos(\theta)$ and $\tan(\theta)$

**Problem 5**: Simplify: $\sin^2(x) + \cos^2(x) + \tan^2(x)$

### Intermediate Level

**Problem 6**: Solve $2\cos(x) + \sqrt{3} = 0$ for $x \in [0, 2\pi)$

**Problem 7**: Find exact value of $\cos(15°)$ using sum/difference formulas

**Problem 8**: If $\cos(\theta) = -\frac{3}{5}$ and $\pi < \theta < \frac{3\pi}{2}$, find $\sin(2\theta)$

**Problem 9**: Prove: $\frac{\sin(x)}{1 - \cos(x)} = \csc(x) + \cot(x)$

**Problem 10**: Solve $\tan^2(x) - 3 = 0$ for $x \in [0, 2\pi)$

### Advanced Level

**Problem 11**: Solve $\sin(x) + \sin(2x) = 0$ for $x \in [0, 2\pi)$

**Problem 12**: Prove: $\cos^4(x) - \sin^4(x) = \cos(2x)$

**Problem 13**: Find all solutions: $2\sin^2(x) + 3\cos(x) = 0$

**Problem 14**: A weight attached to a spring oscillates with displacement $d(t) = 5\cos(4t) + 3\sin(4t)$ cm. Express in form $d(t) = A\cos(4t - \phi)$ and find amplitude and phase shift.

**Problem 15**: Given vectors $\mathbf{u} = (3, 4)$ and $\mathbf{v} = (-1, 2)$, find the angle between them using cosine similarity.

---

## Summary and Key Takeaways

### Core Concepts Mastered

1. **Angle Measurement**
   - Degree and radian conversions
   - Radian as natural unit for calculus

2. **Unit Circle**
   - Foundation for all trig functions
   - $(\cos\theta, \sin\theta)$ as coordinates

3. **Six Trig Functions**
   - Primary: sine, cosine, tangent
   - Reciprocal: cosecant, secant, cotangent
   - Domains, ranges, periods

4. **Fundamental Identities**
   - Pythagorean: $\sin^2 + \cos^2 = 1$
   - Reciprocal, quotient, even-odd

5. **Advanced Identities**
   - Sum/difference formulas
   - Double angle, half angle
   - Product-to-sum, sum-to-product

6. **Inverse Functions**
   - $\arcsin$, $\arccos$, $\arctan$
   - Domain and range restrictions

7. **Solving Equations**
   - Reference angles
   - Multiple solutions using periodicity

8. **Applications**
   - Periodic phenomena modeling
   - Fourier analysis foundations
   - Cosine similarity (ML)
   - Rotation matrices
   - Wave superposition

### Essential Formulas

| Category | Formula |
|----------|---------|
| **Pythagorean** | $\sin^2\theta + \cos^2\theta = 1$ |
| **Sum** | $\sin(\alpha+\beta) = \sin\alpha\cos\beta + \cos\alpha\sin\beta$ |
| **Double Angle** | $\sin(2\theta) = 2\sin\theta\cos\theta$ |
| **Cosine Double** | $\cos(2\theta) = \cos^2\theta - \sin^2\theta$ |
| **Inverse** | $\sin(\arcsin(x)) = x$ for $x \in [-1,1]$ |

### Connections to Other Topics

- **Week 5 (Functions)**: Trig functions are periodic functions
- **Week 6 (Exponentials)**: Euler's formula: $e^{i\theta} = \cos\theta + i\sin\theta$
- **Future (Calculus)**: Derivatives of trig functions
- **Signal Processing**: Fourier transforms decompose signals
- **Machine Learning**: Activation functions, optimization, embeddings
- **Linear Algebra**: Rotations, orthogonal transformations

### Study Checklist

- [ ] Can convert between degrees and radians
- [ ] Know unit circle values for special angles
- [ ] Can evaluate six trig functions
- [ ] Master Pythagorean identity
- [ ] Understand even-odd properties
- [ ] Can apply sum/difference formulas
- [ ] Know double angle formulas
- [ ] Understand inverse trig functions
- [ ] Can solve basic trig equations
- [ ] Recognize periodic patterns in data
- [ ] Understand Fourier analysis concept
- [ ] Know cosine similarity application
- [ ] Avoid common pitfalls (radian vs degree, notation)

---

## Additional Resources

### Recommended Reading

1. **Textbook**: IIT Madras BSMA1001 Week 7 materials
2. **Interactive**: Unit Circle visualization tools
3. **Practice**: Khan Academy - Trigonometry
4. **Applications**: "Digital Signal Processing" by Oppenheim & Schafer

### Online Tools

- **Desmos**: Graph trig functions with transformations
- **GeoGebra**: Interactive unit circle
- **WolframAlpha**: Solve trig equations and identities

### Next Week Preview

**Week 8: Sequences and Series**
- Arithmetic and geometric sequences
- Summation notation and properties
- Convergence tests
- Taylor series introduction
- Applications to algorithms and complexity

---

**End of Week 7 Notes**

*These notes are part of the IIT Madras BS in Data Science Foundation Level coursework.*
