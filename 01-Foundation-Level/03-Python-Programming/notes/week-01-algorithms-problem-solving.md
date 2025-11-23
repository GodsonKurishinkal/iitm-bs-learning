# Week 1: Introduction to Algorithms and Problem Solving

---
**Date**: 2025-11-23
**Course**: BSCS1002 - Programming in Python
**Level**: Foundation
**Week**: 1 of 12
**Source**: IIT Madras Python Programming Week 1
**Topic Area**: Algorithms, Problem Solving, Computational Thinking
**Tags**: #BSCS1002 #Algorithms #ProblemSolving #Python #Week1 #Foundation
---

## 📋 Bottom Line Up Front (BLUF)

**What**: Algorithms are step-by-step procedures for solving problems, forming the foundation of all programming and computational thinking.

**Why**: Before writing code, you must understand the problem and design a clear solution strategy - algorithms provide this systematic approach to problem-solving in data science and software engineering.

**Key Takeaway**: Mastering algorithm design (breaking problems into clear steps) is more important than memorizing syntax - good algorithms translate into efficient, maintainable code in any programming language.

---

## 🎯 Learning Objectives

By the end of this week, you should be able to:

1. ✅ Define what an algorithm is and explain its key characteristics
2. ✅ Break down real-world problems into clear, step-by-step algorithmic solutions
3. ✅ Distinguish between well-defined and ambiguous problem statements
4. ✅ Write algorithms in plain English (pseudocode) before coding
5. ✅ Evaluate algorithms for correctness and identify edge cases
6. ✅ Translate simple algorithms into Python code

---

## 📚 Table of Contents

1. [What is an Algorithm?](#what-is-an-algorithm)
2. [Characteristics of Good Algorithms](#characteristics-of-good-algorithms)
3. [Problem-Solving Framework](#problem-solving-framework)
4. [Algorithm Representation Methods](#algorithm-representation-methods)
5. [From Algorithm to Code](#from-algorithm-to-code)
6. [Common Algorithm Patterns](#common-algorithm-patterns)
7. [Data Science Applications](#data-science-applications)
8. [Common Pitfalls](#common-pitfalls)
9. [Python Implementation](#python-implementation)
10. [Practice Problems](#practice-problems)

---

## 🔍 What is an Algorithm?

### Definition

**Definition:** An **algorithm** is a finite sequence of well-defined instructions for solving a specific problem or performing a computation.

**Etymology:** Named after Persian mathematician Muhammad ibn Musa al-Khwarizmi (9th century), whose work on systematic procedures laid the foundation for modern algorithms.

**Key Properties:**
- **Input:** Zero or more inputs (data provided to the algorithm)
- **Output:** At least one output (result produced)
- **Definiteness:** Each step is precisely defined, unambiguous
- **Finiteness:** Algorithm terminates after a finite number of steps
- **Effectiveness:** Each step is basic enough to be executed

### Everyday Algorithms

Algorithms aren't just for computers - we use them constantly:

**Example 1: Making Tea**
```
Algorithm: MakeTea
Input: Tea bag, hot water, cup, sugar (optional)
Output: Cup of tea

Steps:
1. Place tea bag in cup
2. Pour hot water into cup
3. Wait 3-5 minutes
4. Remove tea bag
5. If sugar desired:
   - Add sugar
   - Stir well
6. Serve
```

**Example 2: Finding Maximum in a List**
```
Algorithm: FindMaximum
Input: A list of numbers [a₁, a₂, ..., aₙ]
Output: The largest number in the list

Steps:
1. Set max = first number in list
2. For each remaining number:
   - If number > max:
     - Set max = number
3. Return max
```

**Example 3: Login Authentication**
```
Algorithm: AuthenticateUser
Input: username, password
Output: "Success" or "Failure"

Steps:
1. Retrieve stored password for username from database
2. If username not found:
   - Return "Failure - user not found"
3. If entered password matches stored password:
   - Return "Success"
4. Else:
   - Return "Failure - incorrect password"
```

### Algorithms vs Programs

| Algorithm | Program |
|-----------|---------|
| Language-independent solution | Language-specific implementation |
| High-level steps | Detailed syntax |
| Abstract representation | Executable code |
| Focus on logic | Focus on syntax and efficiency |

**Example:**

**Algorithm (pseudocode):**
```
To calculate sum of numbers from 1 to n:
1. Set sum = 0
2. For each number from 1 to n:
   - Add number to sum
3. Return sum
```

**Program (Python):**
```python
def calculate_sum(n):
    sum_total = 0
    for number in range(1, n + 1):
        sum_total += number
    return sum_total
```

**Same algorithm, different languages:**
```javascript
// JavaScript
function calculateSum(n) {
    let sum = 0;
    for (let i = 1; i <= n; i++) {
        sum += i;
    }
    return sum;
}
```

---

## ⭐ Characteristics of Good Algorithms

### 1. Correctness

**Definition:** Algorithm produces correct output for all valid inputs, including edge cases.

**Example:** Algorithm for checking if number is prime

❌ **Incorrect Algorithm:**
```
1. For each number from 2 to n-1:
   - If n is divisible by number:
     - Return "Not Prime"
2. Return "Prime"
```

**Problem:** Doesn't handle n ≤ 1 (edge case)

✅ **Correct Algorithm:**
```
1. If n ≤ 1:
   - Return "Not Prime"
2. If n = 2:
   - Return "Prime"
3. For each number from 2 to √n:
   - If n is divisible by number:
     - Return "Not Prime"
4. Return "Prime"
```

### 2. Clarity and Readability

**Definition:** Algorithm steps are easy to understand and follow.

❌ **Unclear:**
```
1. x = a
2. Do stuff
3. Return something
```

✅ **Clear:**
```
Algorithm: CalculateAverage
Input: List of numbers
Output: Average value

1. Set sum = 0
2. Set count = 0
3. For each number in list:
   - Add number to sum
   - Increment count
4. If count = 0:
   - Return "Error: Empty list"
5. Calculate average = sum / count
6. Return average
```

### 3. Efficiency

**Definition:** Algorithm uses minimal computational resources (time, memory).

**Example:** Finding if a number exists in a list

**Approach 1 - Linear Search (Less Efficient):**
```
Time: Check each element one by one
Best case: O(1) - first element
Worst case: O(n) - last element or not found
```

**Approach 2 - Binary Search (More Efficient, for sorted lists):**
```
Time: Divide list in half repeatedly
Best case: O(1) - middle element
Worst case: O(log n) - much faster for large lists
```

**Trade-off:** Binary search requires sorted list (preprocessing cost)

### 4. Generality

**Definition:** Algorithm works for a broad range of inputs, not just specific cases.

❌ **Too Specific:**
```
Algorithm: AddTwoNumbers
1. Return 5 + 7
```

✅ **General:**
```
Algorithm: AddTwoNumbers
Input: a, b (two numbers)
Output: Sum of a and b

1. Calculate sum = a + b
2. Return sum
```

### 5. Robustness

**Definition:** Algorithm handles unexpected or invalid inputs gracefully.

**Example:** Division algorithm

❌ **Not Robust:**
```
def divide(a, b):
    return a / b
```
**Problem:** Crashes when b = 0

✅ **Robust:**
```python
def divide(a, b):
    if b == 0:
        return "Error: Division by zero"
    return a / b
```

---

## 🧩 Problem-Solving Framework

### The 4-Step Approach

**Step 1: Understand the Problem**
- What are the inputs?
- What is the expected output?
- Are there constraints or special cases?
- Can you restate the problem in your own words?

**Step 2: Design the Algorithm**
- Break problem into smaller sub-problems
- Identify patterns from similar problems
- Write step-by-step solution in plain English
- Consider edge cases

**Step 3: Implement the Algorithm**
- Translate pseudocode to programming language
- Use meaningful variable names
- Add comments for complex logic

**Step 4: Test and Refine**
- Test with normal cases
- Test edge cases (empty input, zero, negative numbers)
- Test boundary values
- Optimize if needed

### Example Application: Calculate Student Grade

**Step 1: Understand**
- Input: Student's numerical score (0-100)
- Output: Letter grade (A, B, C, D, F)
- Constraints: Score must be 0-100
- Grading scale:
  - A: 90-100
  - B: 80-89
  - C: 70-79
  - D: 60-69
  - F: 0-59

**Step 2: Design Algorithm**
```
Algorithm: AssignGrade
Input: score (number from 0 to 100)
Output: grade (letter A, B, C, D, or F)

Steps:
1. Validate input:
   - If score < 0 or score > 100:
     - Return "Error: Invalid score"

2. Determine grade:
   - If score >= 90:
     - Set grade = "A"
   - Else if score >= 80:
     - Set grade = "B"
   - Else if score >= 70:
     - Set grade = "C"
   - Else if score >= 60:
     - Set grade = "D"
   - Else:
     - Set grade = "F"

3. Return grade
```

**Step 3: Implement in Python**
```python
def assign_grade(score):
    """
    Assigns letter grade based on numerical score.
    
    Args:
        score (float): Numerical score from 0 to 100
    
    Returns:
        str: Letter grade (A, B, C, D, F) or error message
    """
    # Validate input
    if score < 0 or score > 100:
        return "Error: Invalid score"
    
    # Determine grade
    if score >= 90:
        grade = "A"
    elif score >= 80:
        grade = "B"
    elif score >= 70:
        grade = "C"
    elif score >= 60:
        grade = "D"
    else:
        grade = "F"
    
    return grade
```

**Step 4: Test**
```python
# Test cases
print(assign_grade(95))    # Expected: "A"
print(assign_grade(85))    # Expected: "B"
print(assign_grade(75))    # Expected: "C"
print(assign_grade(65))    # Expected: "D"
print(assign_grade(55))    # Expected: "F"
print(assign_grade(100))   # Expected: "A" (boundary)
print(assign_grade(0))     # Expected: "F" (boundary)
print(assign_grade(-10))   # Expected: Error message
print(assign_grade(150))   # Expected: Error message
```

---

## 📝 Algorithm Representation Methods

### 1. Natural Language (Plain English)

**Advantages:**
- Easy to understand for non-programmers
- No syntax constraints
- Focuses on logic, not implementation

**Disadvantages:**
- Can be ambiguous
- Verbose for complex algorithms

**Example:**
```
To find the average of three numbers:
- Add all three numbers together
- Divide the sum by 3
- Return the result
```

### 2. Pseudocode

**Definition:** Structured representation using programming-like syntax but not tied to specific language.

**Conventions:**
- Use indentation for block structure
- UPPERCASE for keywords (IF, FOR, WHILE)
- Mixed case for variables (totalSum, count)
- Comments with //

**Example:**
```
FUNCTION CalculateAverage(num1, num2, num3)
    // Calculate sum of three numbers
    sum ← num1 + num2 + num3
    
    // Calculate average
    average ← sum / 3
    
    // Return result
    RETURN average
END FUNCTION
```

### 3. Flowcharts

**Definition:** Visual representation using standardized symbols.

**Common Symbols:**
- Oval: Start/End
- Rectangle: Process/Operation
- Diamond: Decision/Condition
- Parallelogram: Input/Output
- Arrow: Flow direction

**Example: Check if number is positive, negative, or zero**

```
        [Start]
           |
      [Input: num]
           |
       <num > 0?>---Yes--->[Print "Positive"]
           |                       |
          No                       |
           |                       |
       <num < 0?>---Yes--->[Print "Negative"]
           |                       |
          No                       |
           |                       |
    [Print "Zero"]                 |
           |                       |
           +-------+-------+-------+
                   |
                 [End]
```

### 4. Structured English

**Example:**
```
Algorithm to calculate factorial of n:
BEGIN
    IF n is less than 0 THEN
        RETURN error message
    END IF
    
    SET result to 1
    FOR i from 1 to n DO
        MULTIPLY result by i
    END FOR
    
    RETURN result
END
```

---

## 💻 From Algorithm to Code

### Translation Process

**Algorithm (Pseudocode):**
```
FUNCTION IsEven(number)
    IF number modulo 2 equals 0 THEN
        RETURN True
    ELSE
        RETURN False
    END IF
END FUNCTION
```

**Python Implementation:**
```python
def is_even(number):
    if number % 2 == 0:
        return True
    else:
        return False
```

**Simplified Python:**
```python
def is_even(number):
    return number % 2 == 0
```

### Example 1: Sum of First N Natural Numbers

**Problem:** Calculate 1 + 2 + 3 + ... + n

**Algorithm:**
```
Algorithm: SumOfNNumbers
Input: n (positive integer)
Output: sum of numbers from 1 to n

1. Initialize sum = 0
2. FOR i = 1 to n:
     Add i to sum
3. RETURN sum
```

**Python Implementation - Method 1 (Loop):**
```python
def sum_of_n_numbers(n):
    """Calculate sum of first n natural numbers using loop."""
    total = 0
    for i in range(1, n + 1):
        total += i
    return total

# Test
print(sum_of_n_numbers(5))  # Output: 15 (1+2+3+4+5)
print(sum_of_n_numbers(10)) # Output: 55
```

**Python Implementation - Method 2 (Formula):**
```python
def sum_of_n_numbers_formula(n):
    """Calculate sum using mathematical formula: n(n+1)/2"""
    return n * (n + 1) // 2

# Test
print(sum_of_n_numbers_formula(5))  # Output: 15
print(sum_of_n_numbers_formula(10)) # Output: 55
```

**Comparison:**
- Method 1: Time complexity O(n) - iterates n times
- Method 2: Time complexity O(1) - constant time
- Both produce same result, Method 2 is more efficient

### Example 2: Find Largest of Three Numbers

**Algorithm:**
```
Algorithm: FindLargest
Input: a, b, c (three numbers)
Output: largest of the three

1. Assume largest = a
2. IF b > largest THEN
     Set largest = b
3. IF c > largest THEN
     Set largest = c
4. RETURN largest
```

**Python Implementation:**
```python
def find_largest(a, b, c):
    """Find largest of three numbers."""
    largest = a
    
    if b > largest:
        largest = b
    
    if c > largest:
        largest = c
    
    return largest

# Test
print(find_largest(5, 10, 3))   # Output: 10
print(find_largest(25, 15, 30)) # Output: 30
print(find_largest(8, 8, 8))    # Output: 8 (all equal)
```

**Alternative (using built-in function):**
```python
def find_largest_builtin(a, b, c):
    """Find largest using Python's max function."""
    return max(a, b, c)
```

---

## 🔄 Common Algorithm Patterns

### 1. Sequential Pattern

**Description:** Execute steps one after another in order.

**Example: Calculate area and perimeter of rectangle**
```python
def rectangle_calculations(length, width):
    """Calculate area and perimeter of rectangle."""
    # Step 1: Calculate area
    area = length * width
    
    # Step 2: Calculate perimeter
    perimeter = 2 * (length + width)
    
    # Step 3: Return both values
    return area, perimeter

# Test
area, perimeter = rectangle_calculations(5, 3)
print(f"Area: {area}, Perimeter: {perimeter}")
# Output: Area: 15, Perimeter: 16
```

### 2. Conditional Pattern (Selection)

**Description:** Execute different steps based on conditions.

**Example: Determine ticket price based on age**
```python
def calculate_ticket_price(age):
    """Calculate ticket price based on age categories."""
    if age < 0:
        return "Error: Invalid age"
    elif age <= 5:
        return 0  # Free for children 5 and under
    elif age <= 12:
        return 10  # Child ticket
    elif age <= 59:
        return 15  # Adult ticket
    else:
        return 12  # Senior discount

# Test
print(calculate_ticket_price(4))   # Output: 0
print(calculate_ticket_price(10))  # Output: 10
print(calculate_ticket_price(30))  # Output: 15
print(calculate_ticket_price(65))  # Output: 12
```

### 3. Iterative Pattern (Repetition)

**Description:** Repeat steps multiple times.

**Example: Calculate factorial (n! = n × (n-1) × ... × 1)**
```python
def factorial(n):
    """Calculate factorial of n."""
    if n < 0:
        return "Error: Factorial not defined for negative numbers"
    
    result = 1
    for i in range(1, n + 1):
        result *= i
    
    return result

# Test
print(factorial(5))  # Output: 120 (5 × 4 × 3 × 2 × 1)
print(factorial(0))  # Output: 1 (by definition)
print(factorial(7))  # Output: 5040
```

### 4. Accumulator Pattern

**Description:** Build up a result by accumulating values.

**Example: Sum all even numbers in a list**
```python
def sum_even_numbers(numbers):
    """Sum all even numbers in a list."""
    total = 0  # Accumulator
    
    for num in numbers:
        if num % 2 == 0:  # Check if even
            total += num  # Accumulate
    
    return total

# Test
print(sum_even_numbers([1, 2, 3, 4, 5, 6]))  # Output: 12 (2+4+6)
print(sum_even_numbers([10, 15, 20, 25]))    # Output: 30 (10+20)
```

### 5. Counter Pattern

**Description:** Count occurrences of specific events or items.

**Example: Count vowels in a string**
```python
def count_vowels(text):
    """Count number of vowels in text."""
    vowels = "aeiouAEIOU"
    count = 0  # Counter
    
    for char in text:
        if char in vowels:
            count += 1  # Increment counter
    
    return count

# Test
print(count_vowels("Hello World"))      # Output: 3 (e, o, o)
print(count_vowels("Data Science"))     # Output: 5 (a, a, i, e, e)
print(count_vowels("Python"))           # Output: 1 (o)
```

### 6. Search Pattern

**Description:** Look for specific item or condition.

**Example: Check if element exists in list**
```python
def contains_element(lst, target):
    """Check if target exists in list."""
    for element in lst:
        if element == target:
            return True  # Found it!
    
    return False  # Not found

# Test
numbers = [10, 20, 30, 40, 50]
print(contains_element(numbers, 30))  # Output: True
print(contains_element(numbers, 25))  # Output: False
```

### 7. Extreme Value Pattern

**Description:** Find maximum or minimum value.

**Example: Find minimum value in list**
```python
def find_minimum(numbers):
    """Find smallest number in list."""
    if not numbers:  # Empty list
        return None
    
    minimum = numbers[0]  # Assume first is minimum
    
    for num in numbers[1:]:  # Check remaining
        if num < minimum:
            minimum = num  # Update minimum
    
    return minimum

# Test
print(find_minimum([5, 2, 9, 1, 7]))  # Output: 1
print(find_minimum([100, -50, 25]))   # Output: -50
```

---

## 📊 Data Science Applications

### Example 1: Data Cleaning - Remove Missing Values

**Problem:** Remove all None values from dataset.

**Algorithm:**
```
Algorithm: RemoveMissing
Input: data_list (list with possible None values)
Output: clean_list (list without None values)

1. Create empty clean_list
2. FOR each item in data_list:
     IF item is not None:
       Add item to clean_list
3. RETURN clean_list
```

**Python Implementation:**
```python
def remove_missing(data):
    """Remove None values from list."""
    clean_data = []
    
    for item in data:
        if item is not None:
            clean_data.append(item)
    
    return clean_data

# Test with sample data
raw_data = [10, None, 25, 30, None, 45, 50]
cleaned = remove_missing(raw_data)
print(cleaned)  # Output: [10, 25, 30, 45, 50]

# Alternative using list comprehension
def remove_missing_compact(data):
    return [item for item in data if item is not None]
```

### Example 2: Feature Engineering - Normalize Values

**Problem:** Scale numerical values to 0-1 range (min-max normalization).

**Formula:** normalized_value = (value - min) / (max - min)

**Algorithm:**
```
Algorithm: NormalizeData
Input: values (list of numbers)
Output: normalized_values (list scaled to 0-1)

1. Find minimum value in list
2. Find maximum value in list
3. Create empty normalized_values list
4. FOR each value in values:
     Calculate normalized = (value - min) / (max - min)
     Add normalized to normalized_values
5. RETURN normalized_values
```

**Python Implementation:**
```python
def normalize_data(values):
    """Normalize values to 0-1 range using min-max scaling."""
    if not values:
        return []
    
    min_val = min(values)
    max_val = max(values)
    
    # Handle case where all values are the same
    if max_val == min_val:
        return [0.5] * len(values)
    
    normalized = []
    for value in values:
        norm_value = (value - min_val) / (max_val - min_val)
        normalized.append(norm_value)
    
    return normalized

# Test
data = [10, 20, 30, 40, 50]
normalized = normalize_data(data)
print(normalized)
# Output: [0.0, 0.25, 0.5, 0.75, 1.0]

# Verify: min becomes 0, max becomes 1
print(f"Min: {min(normalized)}, Max: {max(normalized)}")
```

### Example 3: Model Evaluation - Calculate Accuracy

**Problem:** Calculate classification accuracy (correct predictions / total predictions).

**Algorithm:**
```
Algorithm: CalculateAccuracy
Input: predictions (list), actual_labels (list)
Output: accuracy (percentage)

1. Initialize correct_count = 0
2. FOR i from 0 to length of predictions:
     IF predictions[i] equals actual_labels[i]:
       Increment correct_count
3. Calculate accuracy = correct_count / total_predictions
4. RETURN accuracy
```

**Python Implementation:**
```python
def calculate_accuracy(predictions, actual_labels):
    """
    Calculate classification accuracy.
    
    Args:
        predictions: List of predicted values
        actual_labels: List of actual/true values
    
    Returns:
        float: Accuracy as decimal (0.0 to 1.0)
    """
    if len(predictions) != len(actual_labels):
        return "Error: Lists must have same length"
    
    if len(predictions) == 0:
        return 0.0
    
    correct_count = 0
    
    for pred, actual in zip(predictions, actual_labels):
        if pred == actual:
            correct_count += 1
    
    accuracy = correct_count / len(predictions)
    return accuracy

# Test with sample predictions
predictions = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]
actual =      [1, 0, 1, 0, 0, 1, 1, 0, 1, 1]
#             ✓  ✓  ✓  ✗  ✓  ✓  ✗  ✓  ✓  ✓  = 8/10 correct

accuracy = calculate_accuracy(predictions, actual)
print(f"Accuracy: {accuracy:.2f} ({accuracy*100:.1f}%)")
# Output: Accuracy: 0.80 (80.0%)
```

### Example 4: Data Analysis - Moving Average

**Problem:** Calculate moving average (useful for smoothing time series data).

**Algorithm:**
```
Algorithm: MovingAverage
Input: data (list), window_size (integer)
Output: smoothed_data (list of averages)

1. Create empty smoothed_data list
2. FOR i from 0 to (length of data - window_size):
     Extract window = data[i : i + window_size]
     Calculate average of window
     Add average to smoothed_data
3. RETURN smoothed_data
```

**Python Implementation:**
```python
def moving_average(data, window_size):
    """
    Calculate moving average with specified window size.
    
    Args:
        data: List of numerical values
        window_size: Size of moving window
    
    Returns:
        list: Moving averages
    """
    if window_size > len(data):
        return "Error: Window size larger than data"
    
    averages = []
    
    for i in range(len(data) - window_size + 1):
        window = data[i:i + window_size]
        avg = sum(window) / window_size
        averages.append(avg)
    
    return averages

# Test with sample time series data
sales_data = [100, 120, 110, 130, 125, 140, 135, 150]
ma_3 = moving_average(sales_data, 3)
print("3-day moving average:", [round(x, 1) for x in ma_3])
# Output: [110.0, 120.0, 121.7, 131.7, 133.3, 141.7]

# Visualization of smoothing effect
print("\nOriginal:", sales_data)
print("Smoothed:", [round(x) for x in ma_3])
```

---

## ⚠️ Common Pitfalls

### Pitfall 1: Off-by-One Errors

❌ **Incorrect:**
```python
# Trying to print numbers 1 to 10
for i in range(10):
    print(i)
# Output: 0 1 2 3 4 5 6 7 8 9 (only 0-9!)
```

✅ **Correct:**
```python
for i in range(1, 11):  # 11 is exclusive
    print(i)
# Output: 1 2 3 4 5 6 7 8 9 10
```

**Why it matters:** In data science, off-by-one errors can lead to:
- Missing the last data point in analysis
- Index out of bounds errors
- Incorrect train/test splits

### Pitfall 2: Not Handling Edge Cases

❌ **Incomplete:**
```python
def get_first_element(lst):
    return lst[0]  # Crashes on empty list!

# Error with empty list
print(get_first_element([]))  # IndexError!
```

✅ **Robust:**
```python
def get_first_element(lst):
    if not lst:  # Check if empty
        return None
    return lst[0]

# Now handles edge case
print(get_first_element([]))      # Output: None
print(get_first_element([1, 2]))  # Output: 1
```

**Edge cases to always consider:**
- Empty collections
- Zero values
- Negative numbers
- Very large numbers
- Single-element collections
- Duplicate values

### Pitfall 3: Infinite Loops

❌ **Dangerous:**
```python
# This never stops!
i = 0
while i < 10:
    print(i)
    # Forgot to increment i!
```

✅ **Correct:**
```python
i = 0
while i < 10:
    print(i)
    i += 1  # Remember to update condition variable
```

**Prevention strategy:**
- Always ensure loop condition will eventually become false
- Use for loops when iteration count is known
- Add safety maximum iteration limits for complex conditions

### Pitfall 4: Modifying Collection While Iterating

❌ **Problematic:**
```python
numbers = [1, 2, 3, 4, 5]
for num in numbers:
    if num % 2 == 0:
        numbers.remove(num)  # Don't modify during iteration!
print(numbers)  # Unexpected result: [1, 3, 5]
# 4 was skipped because indices shifted
```

✅ **Correct Approach 1 - Create new list:**
```python
numbers = [1, 2, 3, 4, 5]
odd_numbers = [num for num in numbers if num % 2 != 0]
print(odd_numbers)  # [1, 3, 5]
```

✅ **Correct Approach 2 - Iterate over copy:**
```python
numbers = [1, 2, 3, 4, 5]
for num in numbers[:]:  # Iterate over copy
    if num % 2 == 0:
        numbers.remove(num)
print(numbers)  # [1, 3, 5]
```

### Pitfall 5: Integer Division vs Float Division

❌ **Unexpected (Python 2 behavior):**
```python
# In Python 3, / always returns float
average = 5 / 2
print(average)  # 2.5 (correct in Python 3)

# But // is integer division
average = 5 // 2
print(average)  # 2 (truncates decimal)
```

✅ **Be explicit:**
```python
# For average calculation, use regular division
average = sum(numbers) / len(numbers)

# For integer division (e.g., grouping items), use //
groups = total_items // items_per_group
```

---

## 🐍 Python Implementation

### Complete Example: Student Grade Calculator

**Problem:** Calculate final grade from multiple assessment scores.

**Requirements:**
- Handle multiple assessment types (exams, assignments, projects)
- Apply weighted percentages
- Validate all inputs
- Handle edge cases

**Algorithm:**
```
Algorithm: CalculateFinalGrade
Input: exam_scores (list), assignment_scores (list), project_score (number)
       weights = {exams: 0.5, assignments: 0.3, projects: 0.2}
Output: final_grade (number), letter_grade (string)

1. Validate inputs:
   - Check all lists are non-empty
   - Check all scores are 0-100
   
2. Calculate component averages:
   - exam_avg = average of exam_scores
   - assignment_avg = average of assignment_scores
   - project_avg = project_score
   
3. Calculate weighted final grade:
   - final = (exam_avg × 0.5) + (assignment_avg × 0.3) + (project_avg × 0.2)
   
4. Determine letter grade based on final score

5. Return final_grade, letter_grade
```

**Python Implementation:**
```python
def calculate_final_grade(exam_scores, assignment_scores, project_score):
    """
    Calculate final weighted grade for a student.
    
    Args:
        exam_scores: List of exam scores (0-100)
        assignment_scores: List of assignment scores (0-100)
        project_score: Project score (0-100)
    
    Returns:
        tuple: (final_grade, letter_grade) or error message
    """
    # Validation
    if not exam_scores or not assignment_scores:
        return "Error: Missing scores"
    
    # Validate score ranges
    all_scores = exam_scores + assignment_scores + [project_score]
    for score in all_scores:
        if score < 0 or score > 100:
            return f"Error: Invalid score {score}"
    
    # Calculate component averages
    exam_avg = sum(exam_scores) / len(exam_scores)
    assignment_avg = sum(assignment_scores) / len(assignment_scores)
    project_avg = project_score
    
    # Calculate weighted final grade
    final_grade = (exam_avg * 0.5) + (assignment_avg * 0.3) + (project_avg * 0.2)
    
    # Determine letter grade
    if final_grade >= 90:
        letter = 'A'
    elif final_grade >= 80:
        letter = 'B'
    elif final_grade >= 70:
        letter = 'C'
    elif final_grade >= 60:
        letter = 'D'
    else:
        letter = 'F'
    
    return round(final_grade, 2), letter


# Test cases
print("Test 1: Strong performance")
result = calculate_final_grade([95, 92, 88], [90, 85, 95], 92)
print(f"Final: {result[0]}, Grade: {result[1]}")
# Expected: ~91.1, A

print("\nTest 2: Mixed performance")
result = calculate_final_grade([75, 80, 70], [85, 78, 82], 88)
print(f"Final: {result[0]}, Grade: {result[1]}")
# Expected: ~78.8, C

print("\nTest 3: Edge case - single exam")
result = calculate_final_grade([100], [95, 90], 85)
print(f"Final: {result[0]}, Grade: {result[1]}")
# Expected: ~95.5, A

print("\nTest 4: Error case - invalid score")
result = calculate_final_grade([85, 110], [90], 85)
print(result)
# Expected: Error message

print("\nTest 5: Error case - empty list")
result = calculate_final_grade([], [90], 85)
print(result)
# Expected: Error message
```

**Output:**
```
Test 1: Strong performance
Final: 91.1, Grade: A

Test 2: Mixed performance
Final: 78.8, Grade: C

Test 3: Edge case - single exam
Final: 95.5, Grade: A

Test 4: Error case - invalid score
Error: Invalid score 110

Test 5: Error case - empty list
Error: Missing scores
```

---

## 📝 Practice Problems

### Basic Level

**Problem 1: Temperature Converter**
Write an algorithm and Python function to convert Celsius to Fahrenheit.
Formula: F = (C × 9/5) + 32

<details>
<summary>Solution</summary>

```python
def celsius_to_fahrenheit(celsius):
    """Convert Celsius to Fahrenheit."""
    fahrenheit = (celsius * 9/5) + 32
    return fahrenheit

# Test
print(celsius_to_fahrenheit(0))    # 32°F
print(celsius_to_fahrenheit(100))  # 212°F
print(celsius_to_fahrenheit(25))   # 77°F
```
</details>

**Problem 2: Even or Odd**
Write an algorithm to determine if a number is even or odd.

<details>
<summary>Solution</summary>

```python
def check_even_odd(number):
    """Determine if number is even or odd."""
    if number % 2 == 0:
        return "Even"
    else:
        return "Odd"

# Test
print(check_even_odd(10))  # Even
print(check_even_odd(7))   # Odd
print(check_even_odd(0))   # Even
```
</details>

**Problem 3: Sum of List**
Write an algorithm to calculate the sum of all numbers in a list.

<details>
<summary>Solution</summary>

```python
def sum_list(numbers):
    """Calculate sum of all numbers in list."""
    if not numbers:
        return 0
    
    total = 0
    for num in numbers:
        total += num
    return total

# Test
print(sum_list([1, 2, 3, 4, 5]))  # 15
print(sum_list([10, -5, 3]))      # 8
print(sum_list([]))               # 0
```
</details>

### Intermediate Level

**Problem 4: Count Specific Element**
Write an algorithm to count how many times a specific element appears in a list.

<details>
<summary>Solution</summary>

```python
def count_element(lst, target):
    """Count occurrences of target in list."""
    count = 0
    for element in lst:
        if element == target:
            count += 1
    return count

# Test
numbers = [1, 2, 3, 2, 4, 2, 5]
print(count_element(numbers, 2))  # 3
print(count_element(numbers, 6))  # 0

words = ["apple", "banana", "apple", "cherry", "apple"]
print(count_element(words, "apple"))  # 3
```
</details>

**Problem 5: Palindrome Checker**
Write an algorithm to check if a string is a palindrome (reads same forwards and backwards).

<details>
<summary>Solution</summary>

```python
def is_palindrome(text):
    """Check if string is a palindrome."""
    # Remove spaces and convert to lowercase
    cleaned = text.replace(" ", "").lower()
    
    # Compare with reverse
    return cleaned == cleaned[::-1]

# Test
print(is_palindrome("racecar"))      # True
print(is_palindrome("hello"))        # False
print(is_palindrome("A man a plan a canal Panama"))  # True (ignoring spaces)
```
</details>

**Problem 6: Find Second Largest**
Write an algorithm to find the second largest number in a list.

<details>
<summary>Solution</summary>

```python
def find_second_largest(numbers):
    """Find second largest number in list."""
    if len(numbers) < 2:
        return "Error: Need at least 2 numbers"
    
    # Remove duplicates and sort
    unique_nums = list(set(numbers))
    
    if len(unique_nums) < 2:
        return "Error: All numbers are the same"
    
    unique_nums.sort(reverse=True)
    return unique_nums[1]

# Test
print(find_second_largest([5, 2, 9, 1, 7]))     # 7
print(find_second_largest([10, 10, 5, 5, 3]))   # 5
print(find_second_largest([8]))                 # Error
```
</details>

### Advanced Level

**Problem 7: Prime Number Checker**
Write an algorithm to determine if a number is prime.

<details>
<summary>Solution</summary>

```python
def is_prime(n):
    """Check if number is prime."""
    if n <= 1:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    
    # Check odd divisors up to √n
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    
    return True

# Test
print(is_prime(17))   # True
print(is_prime(20))   # False
print(is_prime(2))    # True
print(is_prime(1))    # False
print(is_prime(97))   # True
```
</details>

**Problem 8: Merge Sorted Lists**
Write an algorithm to merge two sorted lists into one sorted list.

<details>
<summary>Solution</summary>

```python
def merge_sorted_lists(list1, list2):
    """Merge two sorted lists into one sorted list."""
    merged = []
    i, j = 0, 0
    
    # Compare elements from both lists
    while i < len(list1) and j < len(list2):
        if list1[i] <= list2[j]:
            merged.append(list1[i])
            i += 1
        else:
            merged.append(list2[j])
            j += 1
    
    # Add remaining elements
    merged.extend(list1[i:])
    merged.extend(list2[j:])
    
    return merged

# Test
list1 = [1, 3, 5, 7]
list2 = [2, 4, 6, 8]
print(merge_sorted_lists(list1, list2))
# Output: [1, 2, 3, 4, 5, 6, 7, 8]

list3 = [1, 5, 9]
list4 = [2, 3, 4, 10]
print(merge_sorted_lists(list3, list4))
# Output: [1, 2, 3, 4, 5, 9, 10]
```
</details>

**Problem 9: Data Science - Calculate Median**
Write an algorithm to calculate the median of a list of numbers.

<details>
<summary>Solution</summary>

```python
def calculate_median(numbers):
    """Calculate median of list of numbers."""
    if not numbers:
        return None
    
    # Sort the list
    sorted_nums = sorted(numbers)
    n = len(sorted_nums)
    
    # If odd length, return middle element
    if n % 2 == 1:
        return sorted_nums[n // 2]
    
    # If even length, return average of two middle elements
    else:
        mid1 = sorted_nums[n // 2 - 1]
        mid2 = sorted_nums[n // 2]
        return (mid1 + mid2) / 2

# Test
print(calculate_median([5, 2, 8, 1, 9]))      # 5 (middle of 1,2,5,8,9)
print(calculate_median([10, 20, 30, 40]))     # 25 (average of 20,30)
print(calculate_median([7]))                   # 7
print(calculate_median([3, 1, 4, 1, 5, 9]))   # 3.5
```
</details>

---

## 📚 Additional Resources

### Recommended Reading
- **"Python for Everybody"** by Charles Severance - Chapter 1: Introduction
- **"Introduction to Algorithms"** by Cormen et al. - Chapter 1: Foundations
- **"Think Python"** by Allen Downey - Chapter 3: Functions

### Online Resources
- [Python.org Official Tutorial](https://docs.python.org/3/tutorial/)
- [Visualize Python Execution](https://pythontutor.com/)
- [LeetCode Easy Problems](https://leetcode.com/) - For algorithm practice

### Practice Platforms
- **HackerRank** - Python basics track
- **Codewars** - 8 kyu (beginner) problems
- **Project Euler** - Mathematical/computational problems

### Next Steps
- Practice writing algorithms in pseudocode before coding
- Solve 3-5 simple problems daily
- Review code for correctness and edge cases
- Learn to trace algorithm execution step-by-step

---

**Next Week**: [Week 2: Conditionals and Control Flow](week-02-conditionals-control-flow.md)

**Related Topics**:
- [Week 3: Iterations and Loops](week-03-iterations-loops.md)
- [Week 12: Final Project Integration](week-12-final-project.md)
