# Week 4: Ranges and Sequences in Python

---
**Date**: 2025-11-23
**Course**: BSCS1002 - Programming in Python
**Level**: Foundation
**Week**: 4 of 12
**Source**: IIT Madras Python Programming Week 4
**Topic Area**: Sequences, Ranges, Indexing, Slicing
**Tags**: #BSCS1002 #Sequences #Ranges #Indexing #Slicing #Week4 #Foundation
---

## 📋 Bottom Line Up Front (BLUF)

**What**: Sequences are ordered collections in Python (strings, lists, tuples, ranges) that support indexing, slicing, and common operations - the foundation for data manipulation in data science.

**Why**: Understanding sequence operations is critical for data preprocessing, feature extraction, and time series analysis - virtually all data science work involves manipulating sequential data.

**Key Takeaway**: Master indexing (accessing elements), slicing (extracting subsequences), and the range() function to efficiently work with ordered data - these are fundamental building blocks for pandas, numpy, and all data analysis tasks.

---

## 🎯 Learning Objectives

By the end of this week, you should be able to:

1. ✅ Understand what sequences are and identify common sequence types
2. ✅ Use positive and negative indexing to access sequence elements
3. ✅ Apply slicing to extract subsequences with [start:stop:step] notation
4. ✅ Master the range() function for generating number sequences
5. ✅ Perform common sequence operations (concatenation, repetition, membership)
6. ✅ Apply sequence operations to data science problems

---

## 📚 Table of Contents

1. [Understanding Sequences](#understanding-sequences)
2. [Indexing: Accessing Elements](#indexing-accessing-elements)
3. [Slicing: Extracting Subsequences](#slicing-extracting-subsequences)
4. [The Range Function](#the-range-function)
5. [Common Sequence Operations](#common-sequence-operations)
6. [Data Science Applications](#data-science-applications)
7. [Common Pitfalls](#common-pitfalls)
8. [Practice Problems](#practice-problems)

---

## 📦 Understanding Sequences

### What is a Sequence?

**Definition:** A **sequence** is an ordered collection of items where each item has a specific position (index).

**Common sequence types in Python:**

| Type | Example | Mutable? | Use Case |
|------|---------|----------|----------|
| **String** | \`"Python"\` | ❌ No | Text data |
| **List** | \`[1, 2, 3]\` | ✅ Yes | General collections |
| **Tuple** | \`(1, 2, 3)\` | ❌ No | Fixed collections |
| **Range** | \`range(5)\` | ❌ No | Number sequences |

### Example: Different Sequence Types

\`\`\`python
# String - sequence of characters
text = "Data Science"
print(f"String length: {len(text)}")  # 12

# List - sequence of any objects
numbers = [10, 20, 30, 40, 50]
print(f"List length: {len(numbers)}")  # 5

# Tuple - immutable sequence
coordinates = (10.5, 20.3)
print(f"Tuple length: {len(coordinates)}")  # 2

# Range - sequence of numbers
num_range = range(5)
print(f"Range: {list(num_range)}")  # [0, 1, 2, 3, 4]
\`\`\`

---

## 🔢 Indexing: Accessing Elements

### Positive Indexing (0-based)

\`\`\`
String: "P  y  t  h  o  n"
Index:   0  1  2  3  4  5
\`\`\`

\`\`\`python
word = "Python"
print(word[0])  # 'P' - first character
print(word[5])  # 'n' - last character

scores = [85, 92, 78, 95, 88]
print(scores[0])  # 85 - first element
print(scores[4])  # 88 - last element
\`\`\`

### Negative Indexing (from end)

\`\`\`
String: "P  y  t  h  o  n"
Index:  -6 -5 -4 -3 -2 -1
\`\`\`

\`\`\`python
word = "Python"
print(word[-1])  # 'n' - last character
print(word[-2])  # 'o' - second to last

data = [10, 20, 30, 40, 50]
print(data[-1])  # 50 - last element
print(data[-3])  # 30 - third from end
\`\`\`

---

## ✂️ Slicing: Extracting Subsequences

### Basic Slicing: [start:stop]

\`\`\`python
text = "Data Science"
print(text[0:4])    # 'Data' - indices 0,1,2,3
print(text[5:12])   # 'Science'

numbers = [10, 20, 30, 40, 50]
print(numbers[1:4])  # [20, 30, 40]
\`\`\`

### Slicing with Defaults

\`\`\`python
word = "Python"
print(word[:3])     # 'Pyt' - from start to index 3
print(word[2:])     # 'thon' - from index 2 to end
print(word[:])      # 'Python' - entire string (copy)
\`\`\`

### Slicing with Step: [start:stop:step]

\`\`\`python
numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
print(numbers[::2])     # [0, 2, 4, 6, 8] - every 2nd
print(numbers[1::2])    # [1, 3, 5, 7, 9] - odd numbers
print(numbers[::-1])    # [9, 8, 7, ...] - reversed
\`\`\`

---

## 🔢 The Range Function

### Range Syntax

\`\`\`python
# range(stop)
print(list(range(5)))           # [0, 1, 2, 3, 4]

# range(start, stop)
print(list(range(2, 8)))        # [2, 3, 4, 5, 6, 7]

# range(start, stop, step)
print(list(range(0, 20, 3)))    # [0, 3, 6, 9, 12, 15, 18]
print(list(range(10, 0, -1)))   # [10, 9, 8, ..., 1]
\`\`\`

### Range is Memory Efficient

\`\`\`python
# Range doesn't store all values!
big_range = range(1000000)  # Uses ~48 bytes
# vs list would use ~8MB

# Use in loops
for i in range(1, 11):
    print(f"{i}² = {i**2}")
\`\`\`

---

## 🔗 Common Sequence Operations

### Concatenation (+)

\`\`\`python
# Strings
greeting = "Hello" + " " + "World"  # "Hello World"

# Lists
list1 = [1, 2, 3]
list2 = [4, 5, 6]
combined = list1 + list2  # [1, 2, 3, 4, 5, 6]
\`\`\`

### Repetition (*)

\`\`\`python
print("Hi! " * 3)        # "Hi! Hi! Hi! "
zeros = [0] * 5          # [0, 0, 0, 0, 0]
pattern = (1, 2) * 3     # (1, 2, 1, 2, 1, 2)
\`\`\`

### Membership (in)

\`\`\`python
# Check existence
print('P' in "Python")       # True
print(30 in [10, 20, 30])   # True
print('x' not in "Python")   # True

# Validation
valid_grades = ['A', 'B', 'C', 'D', 'F']
if grade in valid_grades:
    print("Valid")
\`\`\`

### Length, Min, Max, Sum

\`\`\`python
numbers = [45, 12, 78, 23, 67]
print(len(numbers))    # 5
print(min(numbers))    # 12
print(max(numbers))    # 78
print(sum(numbers))    # 225
\`\`\`

---

## 📊 Data Science Applications

### Application 1: Time Series Windows

\`\`\`python
def create_windows(data, window_size):
    """Create sliding windows from time series."""
    windows = []
    for i in range(len(data) - window_size + 1):
        window = data[i:i + window_size]
        windows.append(window)
    return windows

prices = [100, 102, 105, 103, 107, 110]
windows = create_windows(prices, 3)
for window in windows:
    print(f"Window: {window} → Avg: {sum(window)/3:.2f}")
\`\`\`

### Application 2: Train-Test Split

\`\`\`python
def train_test_split(data, train_ratio=0.8):
    """Split data into train and test sets."""
    split_point = int(len(data) * train_ratio)
    return data[:split_point], data[split_point:]

data = list(range(1, 101))
train, test = train_test_split(data)
print(f"Train: {len(train)}, Test: {len(test)}")  # 80, 20
\`\`\`

### Application 3: Downsample Data

\`\`\`python
def downsample(data, factor):
    """Reduce data by taking every nth element."""
    return data[::factor]

sensor_data = list(range(0, 1000, 10))  # 100 readings
downsampled = downsample(sensor_data, 5)  # Keep every 5th
print(f"Original: {len(sensor_data)}, Reduced: {len(downsampled)}")
\`\`\`

### Application 4: Extract Features

\`\`\`python
def parse_date(date_str):
    """Extract year, month, day from YYYY-MM-DD."""
    year = int(date_str[0:4])
    month = int(date_str[5:7])
    day = int(date_str[8:10])
    return year, month, day

date = "2024-11-23"
y, m, d = parse_date(date)
print(f"Year: {y}, Month: {m}, Day: {d}")
\`\`\`

---

## ⚠️ Common Pitfalls

### Pitfall 1: Index vs Position

\`\`\`python
word = "Python"
# Position 1 = Index 0!
first = word[0]  # Correct for 1st character
\`\`\`

### Pitfall 2: String Immutability

\`\`\`python
text = "Python"
# text[0] = 'J'  # Error! Strings immutable
text = 'J' + text[1:]  # Create new string
\`\`\`

### Pitfall 3: Off-by-One Errors

\`\`\`python
numbers = [0, 1, 2, 3, 4]
# Want first 5 elements
print(numbers[0:5])  # Correct! Stop is exclusive
\`\`\`

### Pitfall 4: Shallow Copy with *

\`\`\`python
# Wrong: creates references to same list
matrix = [[0] * 3] * 2
matrix[0][0] = 1  # Changes BOTH rows!

# Correct: creates separate lists
matrix = [[0] * 3 for _ in range(2)]
\`\`\`

---

## 📝 Practice Problems

### Basic Level

**Problem 1: Last N Characters**
\`\`\`python
def last_n(text, n):
    return text[-n:]

print(last_n("Python", 3))  # "hon"
\`\`\`

**Problem 2: Reverse String**
\`\`\`python
def reverse(text):
    return text[::-1]

print(reverse("Data"))  # "ataD"
\`\`\`

### Intermediate Level

**Problem 3: Remove Ends**
\`\`\`python
def remove_ends(lst):
    if len(lst) <= 2:
        return []
    return lst[1:-1]

print(remove_ends([1, 2, 3, 4, 5]))  # [2, 3, 4]
\`\`\`

**Problem 4: Swap Halves**
\`\`\`python
def swap_halves(lst):
    mid = len(lst) // 2
    return lst[mid:] + lst[:mid]

print(swap_halves([1, 2, 3, 4]))  # [3, 4, 1, 2]
\`\`\`

### Advanced Level

**Problem 5: Rolling Mean**
\`\`\`python
def rolling_mean(data, window):
    means = []
    for i in range(len(data) - window + 1):
        window_data = data[i:i+window]
        means.append(sum(window_data) / window)
    return means

data = [10, 15, 20, 25, 30]
print(rolling_mean(data, 3))  # [15.0, 20.0, 25.0]
\`\`\`

**Problem 6: Create Batches**
\`\`\`python
def create_batches(data, batch_size):
    batches = []
    for i in range(0, len(data), batch_size):
        batches.append(data[i:i+batch_size])
    return batches

data = list(range(1, 11))
print(create_batches(data, 3))
# [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10]]
\`\`\`

---

## 📚 Additional Resources

### Key Takeaways
- **Indexing**: [i] for access, negative indices from end
- **Slicing**: [start:stop:step], stop is exclusive
- **Range**: Memory-efficient number sequences
- **Operations**: +, *, in, len(), min(), max(), sum()

### Practice
- Implement data preprocessing pipelines
- Apply to real datasets
- Master slicing for pandas/numpy preparation

---

**Previous**: [Week 3: Iterations and Loops](week-03-iterations-loops.md)
**Next**: [Week 5: Lists and Tuples](week-05-lists-tuples.md)
