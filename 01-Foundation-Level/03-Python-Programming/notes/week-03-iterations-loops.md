# Week 3: Iterations and Loops

---
**Date**: 2025-11-23
**Course**: BSCS1002 - Programming in Python
**Level**: Foundation
**Week**: 3 of 12
**Source**: IIT Madras Python Programming Week 3
**Topic Area**: Control Flow, Iterations, Loops
**Tags**: #BSCS1002 #Loops #Iterations #ForLoops #WhileLoops #Week3 #Foundation
---

## 📋 Bottom Line Up Front (BLUF)

**What**: Loops enable repetitive execution of code blocks, allowing programs to process collections, perform calculations repeatedly, and automate tasks that would otherwise require redundant code.

**Why**: Real-world data science problems involve processing thousands or millions of data points - loops are essential for iterating through datasets, training machine learning models, and performing batch operations efficiently.

**Key Takeaway**: Master `for` loops for iterating over known sequences and `while` loops for condition-based repetition - together they provide complete control over repetitive operations in data processing and analysis.

---

## 🎯 Learning Objectives

By the end of this week, you should be able to:

1. ✅ Understand when and why to use loops in programming
2. ✅ Implement `for` loops to iterate over sequences (lists, strings, ranges)
3. ✅ Implement `while` loops for condition-controlled repetition
4. ✅ Use `break` and `continue` statements to control loop execution
5. ✅ Apply nested loops for multi-dimensional data processing
6. ✅ Choose appropriate loop type for different programming scenarios
7. ✅ Implement common loop patterns in data science applications

---

## 📚 Table of Contents

1. [Why We Need Loops](#why-we-need-loops)
2. [For Loops: Iterate Over Sequences](#for-loops-iterate-over-sequences)
3. [While Loops: Condition-Based Repetition](#while-loops-condition-based-repetition)
4. [Loop Control: Break and Continue](#loop-control-break-and-continue)
5. [Nested Loops](#nested-loops)
6. [Loop Patterns and Idioms](#loop-patterns-and-idioms)
7. [Data Science Applications](#data-science-applications)
8. [Common Pitfalls](#common-pitfalls)
9. [Python Implementation](#python-implementation)
10. [Practice Problems](#practice-problems)

---

## 🔄 Why We Need Loops

### The Problem with Repetitive Code

**Without Loops (Bad):**
```python
# Print numbers 1 to 5 - terrible approach!
print(1)
print(2)
print(3)
print(4)
print(5)

# Calculate sum of list elements - impractical!
numbers = [10, 20, 30, 40, 50]
total = numbers[0] + numbers[1] + numbers[2] + numbers[3] + numbers[4]
```

**Problems:**
- Verbose and error-prone
- Doesn't scale (what if we need 1 to 1000?)
- Hard to maintain
- Fixed length (can't handle variable-sized data)

**With Loops (Good):**
```python
# Print numbers 1 to 5 - elegant!
for i in range(1, 6):
    print(i)

# Calculate sum - works for any list size
numbers = [10, 20, 30, 40, 50]
total = 0
for num in numbers:
    total += num
```

**Benefits:**
- Concise and readable
- Scales to any size
- Easy to maintain
- Flexible for different data sizes

### Real-World Analogy

**Manual Task:** Grading 100 student exams by hand, one at a time
**Loop Equivalent:** Process each exam systematically using same grading rubric

```python
# Pseudocode for grading
for each_exam in all_exams:
    grade = calculate_score(each_exam)
    record_grade(grade)
```

---

## 🔁 For Loops: Iterate Over Sequences

### Basic For Loop Syntax

**Structure:**
```python
for variable in sequence:
    # Code to execute for each item
    # Indentation matters!
```

**Components:**
- `for`: Keyword starting the loop
- `variable`: Takes value of each item (one at a time)
- `in`: Keyword
- `sequence`: Collection to iterate over (list, string, range, etc.)
- `:`: Marks start of loop body
- Indented block: Code executed each iteration

### Example 1: Iterating Over a List

```python
fruits = ["apple", "banana", "cherry", "date"]

for fruit in fruits:
    print(f"I like {fruit}")

# Output:
# I like apple
# I like banana
# I like cherry
# I like date
```

**How it works:**
1. First iteration: `fruit = "apple"` → execute body
2. Second iteration: `fruit = "banana"` → execute body
3. Third iteration: `fruit = "cherry"` → execute body
4. Fourth iteration: `fruit = "date"` → execute body
5. Loop ends (no more items)

### Example 2: Iterating Over Strings

**Strings are sequences of characters:**
```python
word = "Python"

for letter in word:
    print(letter)

# Output:
# P
# y
# t
# h
# o
# n
```

**Practical Use - Count Vowels:**
```python
text = "Data Science"
vowels = "aeiouAEIOU"
count = 0

for char in text:
    if char in vowels:
        count += 1

print(f"Number of vowels: {count}")
# Output: Number of vowels: 5 (a, a, i, e, e)
```

### Example 3: Using range() Function

**`range()` generates sequence of numbers:**

**Syntax variations:**
- `range(stop)` → 0 to stop-1
- `range(start, stop)` → start to stop-1
- `range(start, stop, step)` → start to stop-1 with step increment

```python
# Print numbers 0 to 4
for i in range(5):
    print(i)
# Output: 0 1 2 3 4

# Print numbers 1 to 5
for i in range(1, 6):
    print(i)
# Output: 1 2 3 4 5

# Print even numbers 0 to 10
for i in range(0, 11, 2):
    print(i)
# Output: 0 2 4 6 8 10

# Count backwards
for i in range(5, 0, -1):
    print(i)
# Output: 5 4 3 2 1
```

### Example 4: Accessing List with Index

**Using `enumerate()` for index and value:**
```python
colors = ["red", "green", "blue"]

# Method 1: Using range and len
for i in range(len(colors)):
    print(f"Index {i}: {colors[i]}")

# Method 2: Using enumerate (more Pythonic)
for index, color in enumerate(colors):
    print(f"Index {index}: {color}")

# Output (both methods):
# Index 0: red
# Index 1: green
# Index 2: blue
```

**With custom start index:**
```python
for index, color in enumerate(colors, start=1):
    print(f"Color {index}: {color}")

# Output:
# Color 1: red
# Color 2: green
# Color 3: blue
```

### Example 5: Iterating Over Dictionary

```python
student_scores = {
    "Alice": 95,
    "Bob": 87,
    "Charlie": 92
}

# Iterate over keys
for name in student_scores:
    print(name)

# Iterate over values
for score in student_scores.values():
    print(score)

# Iterate over key-value pairs
for name, score in student_scores.items():
    print(f"{name}: {score}")

# Output:
# Alice: 95
# Bob: 87
# Charlie: 92
```

---

## ♾️ While Loops: Condition-Based Repetition

### Basic While Loop Syntax

**Structure:**
```python
while condition:
    # Code to execute while condition is True
    # Must eventually make condition False!
```

**Difference from For Loop:**
- **For loop**: Iterate over known sequence (know how many times)
- **While loop**: Repeat until condition becomes False (don't know how many times)

### Example 6: Basic While Loop

```python
count = 1

while count <= 5:
    print(count)
    count += 1  # CRITICAL: Update condition variable

# Output: 1 2 3 4 5
```

**Execution trace:**
1. Check: `count <= 5`? (1 <= 5) → True → print 1, count becomes 2
2. Check: `count <= 5`? (2 <= 5) → True → print 2, count becomes 3
3. Check: `count <= 5`? (3 <= 5) → True → print 3, count becomes 4
4. Check: `count <= 5`? (4 <= 5) → True → print 4, count becomes 5
5. Check: `count <= 5`? (5 <= 5) → True → print 5, count becomes 6
6. Check: `count <= 5`? (6 <= 5) → False → loop ends

### Example 7: User Input Validation

**Keep asking until valid input:**
```python
password = ""

while len(password) < 8:
    password = input("Enter password (min 8 characters): ")
    if len(password) < 8:
        print("Too short! Try again.")

print("Password accepted!")
```

**Use case**: Unknown number of attempts needed until user provides valid input.

### Example 8: Sum Until Threshold

```python
total = 0
number = 1

while total < 100:
    total += number
    number += 1

print(f"Needed {number-1} numbers to exceed 100")
print(f"Total: {total}")

# Output:
# Needed 14 numbers to exceed 100
# Total: 105 (1+2+3+...+14)
```

### Example 9: Processing Data Stream

**Real-world scenario: reading data until end marker:**
```python
data_stream = [10, 25, 30, -1, 50]  # -1 is end marker
index = 0
total = 0

while data_stream[index] != -1:
    total += data_stream[index]
    index += 1

print(f"Sum before end marker: {total}")
# Output: Sum before end marker: 65
```

### For Loop vs While Loop: When to Use Which?

| Scenario | Use For Loop | Use While Loop |
|----------|-------------|----------------|
| Iterate over collection | ✅ `for item in list` | ❌ More complex |
| Known number of iterations | ✅ `for i in range(10)` | ❌ Unnecessary |
| Condition-based repetition | ❌ Awkward | ✅ `while condition` |
| User input validation | ❌ Don't know iterations | ✅ Until valid |
| Reading until sentinel value | ❌ Don't know when to stop | ✅ Until marker |
| Counting with custom logic | ⚠️ Possible | ✅ More flexible |

**General Rule:**
- **For loop**: When you know what to iterate over
- **While loop**: When you know when to stop (but not how many iterations)

---

## 🎮 Loop Control: Break and Continue

### Break Statement: Exit Loop Early

**Syntax:** `break` - immediately exit the loop, skip remaining iterations

**Example 10: Search for Item**
```python
numbers = [10, 25, 30, 45, 50, 60]
target = 30

for num in numbers:
    if num == target:
        print(f"Found {target}!")
        break  # Stop searching once found
    print(f"Checking {num}...")

# Output:
# Checking 10...
# Checking 25...
# Found 30!
```

**Without break (inefficient):**
```python
# Would continue checking 45, 50, 60 unnecessarily
```

**Example 11: Limit Iterations**
```python
# Process maximum 100 items
count = 0
for item in large_dataset:
    process(item)
    count += 1
    if count >= 100:
        break  # Stop after 100
```

### Continue Statement: Skip Current Iteration

**Syntax:** `continue` - skip rest of current iteration, move to next

**Example 12: Skip Even Numbers**
```python
for i in range(1, 11):
    if i % 2 == 0:
        continue  # Skip even numbers
    print(i)

# Output: 1 3 5 7 9 (only odd numbers)
```

**How it works:**
- i=1: Not even, print 1
- i=2: Even, `continue` skips print
- i=3: Not even, print 3
- i=4: Even, `continue` skips print
- ... and so on

**Example 13: Skip Invalid Data**
```python
data = [10, None, 25, -1, 30, None, 45]

total = 0
for value in data:
    if value is None or value < 0:
        continue  # Skip invalid values
    total += value

print(f"Sum of valid data: {total}")
# Output: Sum of valid data: 110 (10+25+30+45)
```

### Break vs Continue Comparison

```python
# Example: Process numbers, skip negatives, stop at zero
numbers = [5, 10, -3, 15, 0, 20, 25]

print("With CONTINUE:")
for num in numbers:
    if num < 0:
        continue  # Skip negative
    if num == 0:
        break  # Stop at zero
    print(num)

# Output: 5 10 15
```

**Trace:**
- 5: positive → print 5
- 10: positive → print 10
- -3: negative → `continue` (skip)
- 15: positive → print 15
- 0: equals zero → `break` (stop loop)
- 20, 25: never reached

### Else Clause with Loops

**Python special feature: `else` with loops**

**Executes if loop completes normally (not broken):**
```python
# Search for prime number
for num in range(2, 10):
    if num > 7:
        print(f"Found number > 7: {num}")
        break
else:
    print("No number > 7 found")  # Only if no break

# With break: prints "Found number > 7: 8"
# Without break: prints "No number > 7 found"
```

**Use case: Search operations**
```python
def find_in_list(lst, target):
    for item in lst:
        if item == target:
            print(f"Found {target}")
            break
    else:
        print(f"{target} not found")

find_in_list([1, 2, 3, 4], 3)  # Found 3
find_in_list([1, 2, 3, 4], 5)  # 5 not found
```

---

## 🔲 Nested Loops

### Definition

**Nested loop**: A loop inside another loop

**Syntax:**
```python
for outer_var in outer_sequence:
    for inner_var in inner_sequence:
        # Inner loop body (executes for each combination)
    # Outer loop body (executes after inner loop completes)
```

**Execution pattern:**
- For each iteration of outer loop
  - Complete all iterations of inner loop

### Example 14: Multiplication Table

```python
# 5x5 multiplication table
for i in range(1, 6):
    for j in range(1, 6):
        product = i * j
        print(f"{i} × {j} = {product:2d}", end="  ")
    print()  # New line after each row

# Output:
# 1 × 1 =  1  1 × 2 =  2  1 × 3 =  3  1 × 4 =  4  1 × 5 =  5  
# 2 × 1 =  2  2 × 2 =  4  2 × 3 =  6  2 × 4 =  8  2 × 5 = 10  
# 3 × 1 =  3  3 × 2 =  6  3 × 3 =  9  3 × 4 = 12  3 × 5 = 15  
# 4 × 1 =  4  4 × 2 =  8  4 × 3 = 12  4 × 4 = 16  4 × 5 = 20  
# 5 × 1 =  5  5 × 2 = 10  5 × 3 = 15  5 × 4 = 20  5 × 5 = 25
```

**Iteration count:** 5 × 5 = 25 total iterations

### Example 15: Process 2D Grid (Matrix)

```python
# Temperature data: rows=days, columns=times
temperatures = [
    [20, 22, 25, 23],  # Day 1
    [19, 21, 24, 22],  # Day 2
    [21, 23, 26, 24]   # Day 3
]

# Calculate average temperature each day
for day_num, day_temps in enumerate(temperatures, 1):
    total = 0
    for temp in day_temps:
        total += temp
    avg = total / len(day_temps)
    print(f"Day {day_num} average: {avg:.1f}°C")

# Output:
# Day 1 average: 22.5°C
# Day 2 average: 21.5°C
# Day 3 average: 23.5°C
```

### Example 16: Pattern Printing

```python
# Print right triangle pattern
for i in range(1, 6):
    for j in range(i):
        print("*", end="")
    print()

# Output:
# *
# **
# ***
# ****
# *****
```

**How it works:**
- i=1: inner loop runs 1 time → print *
- i=2: inner loop runs 2 times → print **
- i=3: inner loop runs 3 times → print ***
- etc.

### Example 17: Compare Two Lists (Cartesian Product)

```python
colors = ["red", "blue"]
sizes = ["S", "M", "L"]

# Generate all product combinations
for color in colors:
    for size in sizes:
        print(f"{color}-{size}")

# Output:
# red-S
# red-M
# red-L
# blue-S
# blue-M
# blue-L
```

**Total combinations:** 2 colors × 3 sizes = 6

### Performance Consideration

**Time complexity of nested loops:**
```python
# O(n²) - Quadratic time
for i in range(n):
    for j in range(n):
        # Code here
```

**Beware:**
- `n=100`: 10,000 iterations
- `n=1000`: 1,000,000 iterations
- `n=10000`: 100,000,000 iterations (slow!)

**Use nested loops when necessary, but optimize when possible.**

---

## 🧩 Loop Patterns and Idioms

### Pattern 1: Accumulator

**Sum all values:**
```python
numbers = [10, 20, 30, 40, 50]
total = 0  # Accumulator

for num in numbers:
    total += num  # Accumulate

print(f"Total: {total}")  # 150
```

### Pattern 2: Counter

**Count occurrences:**
```python
text = "data science"
count = 0  # Counter

for char in text:
    if char == 'a':
        count += 1

print(f"Letter 'a' appears {count} times")  # 2
```

### Pattern 3: Filter

**Collect items meeting condition:**
```python
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
evens = []  # Filtered list

for num in numbers:
    if num % 2 == 0:
        evens.append(num)

print(evens)  # [2, 4, 6, 8, 10]
```

### Pattern 4: Transform (Map)

**Apply transformation to each item:**
```python
temps_celsius = [0, 10, 20, 30, 40]
temps_fahrenheit = []

for c in temps_celsius:
    f = (c * 9/5) + 32
    temps_fahrenheit.append(f)

print(temps_fahrenheit)  # [32.0, 50.0, 68.0, 86.0, 104.0]
```

### Pattern 5: Find Maximum/Minimum

```python
numbers = [45, 12, 78, 23, 67, 34]

max_val = numbers[0]  # Assume first is max
for num in numbers[1:]:
    if num > max_val:
        max_val = num

print(f"Maximum: {max_val}")  # 78
```

### Pattern 6: Pairwise Processing

**Compare adjacent elements:**
```python
prices = [100, 105, 103, 108, 110]

print("Price changes:")
for i in range(len(prices) - 1):
    change = prices[i+1] - prices[i]
    print(f"Day {i} to {i+1}: {change:+d}")

# Output:
# Day 0 to 1: +5
# Day 1 to 2: -2
# Day 2 to 3: +5
# Day 3 to 4: +2
```

### Pattern 7: Parallel Iteration (zip)

**Iterate multiple lists simultaneously:**
```python
names = ["Alice", "Bob", "Charlie"]
scores = [95, 87, 92]

for name, score in zip(names, scores):
    print(f"{name}: {score}")

# Output:
# Alice: 95
# Bob: 87
# Charlie: 92
```

---

## 📊 Data Science Applications

### Application 1: Calculate Mean (Average)

```python
def calculate_mean(data):
    """Calculate arithmetic mean of dataset."""
    if not data:
        return None
    
    total = 0
    count = 0
    
    for value in data:
        total += value
        count += 1
    
    return total / count

# Test
sales_data = [120, 135, 128, 142, 131, 139, 145]
mean = calculate_mean(sales_data)
print(f"Average sales: ${mean:.2f}")
# Output: Average sales: $134.29
```

### Application 2: Standardize Data (Z-score)

```python
def standardize(data):
    """Standardize data to zero mean, unit variance."""
    # Calculate mean
    mean = sum(data) / len(data)
    
    # Calculate standard deviation
    squared_diffs = 0
    for value in data:
        squared_diffs += (value - mean) ** 2
    variance = squared_diffs / len(data)
    std_dev = variance ** 0.5
    
    # Standardize
    standardized = []
    for value in data:
        z_score = (value - mean) / std_dev
        standardized.append(z_score)
    
    return standardized

# Test
data = [10, 15, 20, 25, 30]
z_scores = standardize(data)
print("Z-scores:", [round(z, 2) for z in z_scores])
# Output: Z-scores: [-1.41, -0.71, 0.0, 0.71, 1.41]
```

### Application 3: Confusion Matrix Metrics

```python
def calculate_metrics(predictions, actuals):
    """Calculate precision, recall from predictions."""
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0
    
    for pred, actual in zip(predictions, actuals):
        if pred == 1 and actual == 1:
            true_positive += 1
        elif pred == 1 and actual == 0:
            false_positive += 1
        elif pred == 0 and actual == 0:
            true_negative += 1
        else:  # pred == 0 and actual == 1
            false_negative += 1
    
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    
    return precision, recall

# Test
preds = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]
actuals = [1, 0, 1, 0, 0, 1, 1, 0, 1, 1]

prec, rec = calculate_metrics(preds, actuals)
print(f"Precision: {prec:.2f}, Recall: {rec:.2f}")
# Output: Precision: 0.83, Recall: 1.00
```

### Application 4: Time Series Smoothing

```python
def exponential_smoothing(data, alpha=0.3):
    """Apply exponential smoothing to time series."""
    if not data:
        return []
    
    smoothed = [data[0]]  # First value unchanged
    
    for i in range(1, len(data)):
        smooth_value = alpha * data[i] + (1 - alpha) * smoothed[i-1]
        smoothed.append(smooth_value)
    
    return smoothed

# Test with noisy data
noisy_data = [100, 110, 105, 120, 115, 125, 130]
smoothed = exponential_smoothing(noisy_data, alpha=0.3)

print("Original:", noisy_data)
print("Smoothed:", [round(x, 1) for x in smoothed])
# Output:
# Original: [100, 110, 105, 120, 115, 125, 130]
# Smoothed: [100.0, 103.0, 103.6, 108.5, 110.5, 114.9, 119.4]
```

---

## ⚠️ Common Pitfalls

### Pitfall 1: Infinite While Loop

❌ **Bug - Never stops:**
```python
count = 1
while count <= 5:
    print(count)
    # Forgot to increment count!
# Prints 1 forever (infinite loop)
```

✅ **Fixed:**
```python
count = 1
while count <= 5:
    print(count)
    count += 1  # Essential!
```

### Pitfall 2: Modifying List While Iterating

❌ **Dangerous:**
```python
numbers = [1, 2, 3, 4, 5]
for num in numbers:
    if num % 2 == 0:
        numbers.remove(num)  # BAD!
# Skips elements due to shifting indices
```

✅ **Safe approach:**
```python
numbers = [1, 2, 3, 4, 5]
odd_numbers = [num for num in numbers if num % 2 != 0]
# Or iterate over copy: for num in numbers[:]:
```

### Pitfall 3: Wrong Range Bounds

❌ **Off-by-one error:**
```python
# Want to process first 5 elements
for i in range(5):  # Correct: 0,1,2,3,4
    print(list[i])

for i in range(1, 5):  # WRONG: 1,2,3,4 (misses index 0, tries index 5)
    print(list[i])
```

### Pitfall 4: Unreachable Break

❌ **Break in wrong place:**
```python
for i in range(10):
    print(i)
break  # WRONG: Outside loop, only executes once anyway
```

✅ **Correct:**
```python
for i in range(10):
    print(i)
    if i == 5:
        break  # Inside loop
```

### Pitfall 5: Nested Loop Complexity

❌ **Inefficient - O(n³):**
```python
# Triple nested loop - very slow for large n
for i in range(n):
    for j in range(n):
        for k in range(n):
            # n³ operations
```

✅ **Optimize when possible:**
```python
# Think if you really need nested loops
# Can some work be done outside inner loops?
```

---

## 📝 Practice Problems

### Basic Level

**Problem 1: Sum of Squares**
Calculate sum of squares of numbers 1 to n.

<details>
<summary>Solution</summary>

```python
def sum_of_squares(n):
    """Calculate 1² + 2² + ... + n²"""
    total = 0
    for i in range(1, n + 1):
        total += i ** 2
    return total

print(sum_of_squares(5))  # 1+4+9+16+25 = 55
```
</details>

**Problem 2: Factorial**
Calculate n! = n × (n-1) × ... × 1

<details>
<summary>Solution</summary>

```python
def factorial(n):
    """Calculate factorial using loop."""
    if n < 0:
        return None
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result

print(factorial(5))  # 120
```
</details>

**Problem 3: Reverse String**
Reverse a string using loop.

<details>
<summary>Solution</summary>

```python
def reverse_string(text):
    """Reverse string using loop."""
    reversed_text = ""
    for char in text:
        reversed_text = char + reversed_text
    return reversed_text

print(reverse_string("Python"))  # nohtyP
```
</details>

### Intermediate Level

**Problem 4: Fibonacci Sequence**
Generate first n Fibonacci numbers.

<details>
<summary>Solution</summary>

```python
def fibonacci(n):
    """Generate first n Fibonacci numbers."""
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    
    fib = [0, 1]
    for i in range(2, n):
        next_fib = fib[i-1] + fib[i-2]
        fib.append(next_fib)
    
    return fib

print(fibonacci(10))
# [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
```
</details>

**Problem 5: Find All Prime Numbers Up to N**

<details>
<summary>Solution</summary>

```python
def find_primes(n):
    """Find all prime numbers up to n."""
    primes = []
    
    for num in range(2, n + 1):
        is_prime = True
        for divisor in range(2, int(num ** 0.5) + 1):
            if num % divisor == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(num)
    
    return primes

print(find_primes(20))
# [2, 3, 5, 7, 11, 13, 17, 19]
```
</details>

**Problem 6: Count Word Frequency**

<details>
<summary>Solution</summary>

```python
def word_frequency(text):
    """Count frequency of each word."""
    words = text.lower().split()
    freq = {}
    
    for word in words:
        if word in freq:
            freq[word] += 1
        else:
            freq[word] = 1
    
    return freq

text = "python is great python is fun"
print(word_frequency(text))
# {'python': 2, 'is': 2, 'great': 1, 'fun': 1}
```
</details>

### Advanced Level

**Problem 7: Matrix Transpose**
Transpose a 2D matrix (swap rows and columns).

<details>
<summary>Solution</summary>

```python
def transpose_matrix(matrix):
    """Transpose matrix using nested loops."""
    rows = len(matrix)
    cols = len(matrix[0])
    
    transposed = []
    for j in range(cols):
        new_row = []
        for i in range(rows):
            new_row.append(matrix[i][j])
        transposed.append(new_row)
    
    return transposed

matrix = [
    [1, 2, 3],
    [4, 5, 6]
]
result = transpose_matrix(matrix)
print(result)
# [[1, 4], [2, 5], [3, 6]]
```
</details>

**Problem 8: Gradient Descent Simulation**

<details>
<summary>Solution</summary>

```python
def gradient_descent(start, learning_rate=0.1, iterations=10):
    """
    Simulate gradient descent on f(x) = x².
    Derivative: f'(x) = 2x
    """
    x = start
    history = [x]
    
    for i in range(iterations):
        gradient = 2 * x  # Derivative of x²
        x = x - learning_rate * gradient
        history.append(x)
    
    return history

# Find minimum of x²
path = gradient_descent(start=10, learning_rate=0.1, iterations=20)
print(f"Start: {path[0]:.4f}")
print(f"End: {path[-1]:.4f}")
print(f"Converges to 0 (minimum of x²)")
```
</details>

---

## 📚 Additional Resources

### Recommended Reading
- "Python for Everybody" - Chapter 5: Iterations
- "Automate the Boring Stuff with Python" - Chapter 2: Flow Control

### Practice Platforms
- **HackerRank**: Python loops challenges
- **LeetCode**: Easy array problems using loops
- **Codewars**: 7-8 kyu loop problems

### Next Steps
- Practice loop problems daily
- Learn list comprehensions (compact loop alternative)
- Study loop optimization techniques
- Apply loops to real datasets

---

**Previous**: [Week 2: Conditionals and Control Flow](week-02-conditionals-control-flow.md)
**Next**: [Week 4: Ranges and Sequences](week-04-ranges-sequences.md)

**Related Topics**:
- [Week 5: Lists and Tuples](week-05-lists-tuples.md)
- [Week 11: Pandas Data Analysis](week-11-pandas-data-analysis.md)
