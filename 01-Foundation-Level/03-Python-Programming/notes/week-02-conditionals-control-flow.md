# Week 2: Conditionals and Control Flow

---
**Date**: 2025-11-23
**Course**: BSCS1002 - Programming in Python
**Level**: Foundation
**Week**: 2 of 12
**Source**: IIT Madras Python Programming Week 2
**Topic Area**: Computer Science, Control Structures, Decision Making
**Tags**: #BSCS1002 #Conditionals #ControlFlow #Python #Week2 #Foundation
---

## 📋 Bottom Line Up Front (BLUF)

**What**: Conditional statements enable programs to make decisions and execute different code paths based on conditions, forming the foundation of program logic and intelligence.

**Why**: Real-world problems require decision-making - from filtering data based on criteria to implementing business rules in machine learning pipelines.

**Key Takeaway**: Mastering `if-elif-else` statements and boolean logic allows you to create dynamic, responsive programs that adapt to different inputs and situations.

---

## 🎯 Learning Objectives

By the end of this week, you should be able to:

1. ✅ Write and evaluate boolean expressions using comparison and logical operators
2. ✅ Implement single and multi-branch conditional statements using if-elif-else
3. ✅ Apply nested conditionals for complex decision-making logic
4. ✅ Use conditional expressions (ternary operator) for concise code
5. ✅ Debug common conditional statement errors and logical pitfalls
6. ✅ Apply conditionals to filter and validate data in data science workflows

---

## 📚 Table of Contents

1. [Boolean Logic and Truth Values](#boolean-logic-and-truth-values)
2. [Comparison Operators](#comparison-operators)
3. [Logical Operators](#logical-operators)
4. [Simple if Statements](#simple-if-statements)
5. [if-else Statements](#if-else-statements)
6. [if-elif-else Chains](#if-elif-else-chains)
7. [Nested Conditionals](#nested-conditionals)
8. [Conditional Expressions (Ternary)](#conditional-expressions-ternary)
9. [Data Science Applications](#data-science-applications)
10. [Common Pitfalls](#common-pitfalls)
11. [Python Implementation Examples](#python-implementation-examples)
12. [Practice Problems](#practice-problems)

---

## ✅ Boolean Logic and Truth Values

### The Boolean Data Type

**Definition:** A boolean is a data type that has only two possible values: `True` or `False`.

**Purpose:** Booleans represent the truth value of logical conditions, enabling programs to make decisions.

**Python Syntax:**
```python
is_raining = True
is_sunny = False

print(type(is_raining))  # <class 'bool'>
```

### Boolean Context

**Key Concept:** In Python, many values can be evaluated in a boolean context:

**Falsy Values** (evaluate to False):
- `False`
- `0`, `0.0`
- `""` (empty string)
- `None`
- `[]`, `{}`, `()` (empty collections)

**Truthy Values** (evaluate to True):
- `True`
- Any non-zero number
- Any non-empty string
- Any non-empty collection

### Example 1: Truth Value Testing

```python
# Explicit boolean
has_permission = True
print(bool(has_permission))  # True

# Numbers
print(bool(0))      # False
print(bool(42))     # True
print(bool(-5))     # True
print(bool(0.0))    # False

# Strings
print(bool(""))          # False (empty)
print(bool("Hello"))     # True
print(bool(" "))         # True (space is a character)

# None
print(bool(None))   # False
```

---

## 🔍 Comparison Operators

### Basic Comparison Operators

Comparison operators compare two values and return a boolean result:

| Operator | Meaning | Example | Result |
|----------|---------|---------|--------|
| `==` | Equal to | `5 == 5` | `True` |
| `!=` | Not equal to | `5 != 3` | `True` |
| `<` | Less than | `3 < 5` | `True` |
| `>` | Greater than | `5 > 3` | `True` |
| `<=` | Less than or equal | `5 <= 5` | `True` |
| `>=` | Greater than or equal | `5 >= 3` | `True` |

### Example 2: Comparison Operations

```python
x = 10
y = 20

# Comparisons
print(x == y)   # False
print(x != y)   # True
print(x < y)    # True
print(x > y)    # False
print(x <= 10)  # True (equal counts)
print(y >= 20)  # True

# String comparisons (lexicographical)
print("apple" < "banana")  # True (alphabetical)
print("Apple" < "apple")   # True (uppercase comes first)
```

### Chaining Comparisons

**Python feature:** Multiple comparisons can be chained elegantly:

```python
x = 15

# Chained comparison
if 10 < x < 20:
    print("x is between 10 and 20")  # Executes

# Equivalent to (but more readable than):
if 10 < x and x < 20:
    print("x is between 10 and 20")

# Multiple chains
age = 25
if 18 <= age <= 65:
    print("Working age")  # Executes
```

### Common Mistake: Assignment vs Comparison

❌ **Wrong:**
```python
x = 5
if x = 10:  # SyntaxError: invalid syntax
    print("x is 10")
```

✅ **Correct:**
```python
x = 5
if x == 10:  # Use == for comparison
    print("x is 10")
```

---

## 🔗 Logical Operators

### The Three Logical Operators

| Operator | Description | Example | Result |
|----------|-------------|---------|--------|
| `and` | True if both are true | `True and True` | `True` |
| `or` | True if at least one is true | `True or False` | `True` |
| `not` | Inverts the boolean value | `not True` | `False` |

### Truth Tables

**AND Truth Table:**
| A | B | A and B |
|---|---|---------|
| True | True | True |
| True | False | False |
| False | True | False |
| False | False | False |

**OR Truth Table:**
| A | B | A or B |
|---|---|--------|
| True | True | True |
| True | False | True |
| False | True | True |
| False | False | False |

**NOT Truth Table:**
| A | not A |
|---|-------|
| True | False |
| False | True |

### Example 3: Logical Operators in Action

```python
age = 25
has_license = True
has_insurance = True

# AND: All conditions must be true
can_drive = age >= 18 and has_license and has_insurance
print(f"Can drive: {can_drive}")  # True

# OR: At least one condition must be true
is_student = False
is_senior = False
gets_discount = is_student or is_senior
print(f"Gets discount: {gets_discount}")  # False

# NOT: Inverts boolean
is_weekend = False
is_weekday = not is_weekend
print(f"Is weekday: {is_weekday}")  # True
```

### Operator Precedence

**Order of evaluation:**
1. `not`
2. `and`
3. `or`

```python
# Without parentheses
result = True or False and False
# Evaluates as: True or (False and False)
print(result)  # True

# With parentheses for clarity
result = (True or False) and False
print(result)  # False
```

### Short-Circuit Evaluation

**Important:** Python stops evaluating as soon as the result is determined.

```python
def expensive_check():
    print("Expensive check called!")
    return True

# AND short-circuit
if False and expensive_check():  # expensive_check() NOT called
    print("Won't execute")

# OR short-circuit
if True or expensive_check():  # expensive_check() NOT called
    print("Will execute")  # Executes
```

---

## 🎯 Simple if Statements

### Basic Syntax

```python
if condition:
    # Code block executes only if condition is True
    statement1
    statement2
```

**Key points:**
- Colon `:` is required after condition
- Indentation (4 spaces) defines the code block
- Multiple statements can be in the block

### Example 4: Simple if Statement

```python
temperature = 35

if temperature > 30:
    print("It's hot outside!")
    print("Remember to drink water.")

print("This always executes")
```

**Output:**
```
It's hot outside!
Remember to drink water.
This always executes
```

### Example 5: Multiple Independent if Statements

```python
score = 85

if score >= 90:
    print("Excellent!")

if score >= 80:
    print("Great job!")  # Executes

if score >= 70:
    print("Good work!")  # Executes

if score >= 60:
    print("You passed!")  # Executes
```

**Output:** All conditions from 80 downward execute!
```
Great job!
Good work!
You passed!
```

---

## ⚖️ if-else Statements

### Syntax

```python
if condition:
    # Executes if condition is True
    statement1
else:
    # Executes if condition is False
    statement2
```

**Purpose:** Provide an alternative path when condition is false.

### Example 6: if-else for Binary Decisions

```python
age = int(input("Enter your age: "))

if age >= 18:
    print("You are an adult.")
    print("You can vote.")
else:
    print("You are a minor.")
    print("You cannot vote yet.")
```

### Example 7: Even/Odd Checker

```python
number = int(input("Enter a number: "))

if number % 2 == 0:
    print(f"{number} is even")
else:
    print(f"{number} is odd")
```

### Example 8: Maximum of Two Numbers

```python
a = 15
b = 23

if a > b:
    maximum = a
else:
    maximum = b

print(f"Maximum: {maximum}")  # Maximum: 23
```

---

## 🔀 if-elif-else Chains

### Syntax

```python
if condition1:
    # Executes if condition1 is True
    statement1
elif condition2:
    # Executes if condition1 is False and condition2 is True
    statement2
elif condition3:
    # Executes if previous conditions False and condition3 True
    statement3
else:
    # Executes if all conditions are False
    statement4
```

**Key point:** Only ONE block executes (the first True condition).

### Example 9: Grade Calculator

```python
score = int(input("Enter your score: "))

if score >= 90:
    grade = "A"
    print("Excellent!")
elif score >= 80:
    grade = "B"
    print("Very good!")
elif score >= 70:
    grade = "C"
    print("Good!")
elif score >= 60:
    grade = "D"
    print("Satisfactory")
else:
    grade = "F"
    print("Needs improvement")

print(f"Grade: {grade}")
```

### Example 10: Traffic Light Simulator

```python
light_color = input("Enter traffic light color: ").lower()

if light_color == "green":
    print("GO")
elif light_color == "yellow":
    print("SLOW DOWN")
elif light_color == "red":
    print("STOP")
else:
    print("Invalid light color!")
```

### Example 11: Season Classifier

```python
month = int(input("Enter month number (1-12): "))

if month in [12, 1, 2]:
    season = "Winter"
elif month in [3, 4, 5]:
    season = "Spring"
elif month in [6, 7, 8]:
    season = "Summer"
elif month in [9, 10, 11]:
    season = "Fall"
else:
    season = "Invalid month"

print(f"Season: {season}")
```

---

## 🪆 Nested Conditionals

### Definition

**Nested conditional:** An if statement inside another if statement.

**Purpose:** Handle complex decision-making with multiple levels of conditions.

### Example 12: Age and Income Eligibility

```python
age = int(input("Enter age: "))
income = float(input("Enter annual income: "))

if age >= 18:
    print("You are an adult.")
    
    if income >= 30000:
        print("You qualify for a premium credit card.")
    else:
        print("You qualify for a standard credit card.")
else:
    print("You are a minor.")
    print("You are not eligible for a credit card.")
```

### Example 13: Triangle Type Classifier

```python
side1 = float(input("Enter side 1: "))
side2 = float(input("Enter side 2: "))
side3 = float(input("Enter side 3: "))

# Check if valid triangle
if side1 + side2 > side3 and side2 + side3 > side1 and side1 + side3 > side2:
    print("Valid triangle")
    
    # Classify type
    if side1 == side2 == side3:
        print("Equilateral triangle")
    elif side1 == side2 or side2 == side3 or side1 == side3:
        print("Isosceles triangle")
    else:
        print("Scalene triangle")
else:
    print("Not a valid triangle!")
```

### Example 14: Nested Grade with Effort Level

```python
score = int(input("Enter score: "))
effort = input("Enter effort level (high/medium/low): ").lower()

if score >= 70:
    print("Passing grade!")
    
    if effort == "high":
        print("Excellent work ethic!")
    elif effort == "medium":
        print("Good effort, keep it up!")
    else:
        print("Try to put in more effort next time.")
else:
    print("Failing grade.")
    
    if effort == "high":
        print("Your effort is noticed. Keep trying!")
    else:
        print("You need to work harder and smarter.")
```

### Best Practice: Avoid Deep Nesting

❌ **Hard to read (too much nesting):**
```python
if condition1:
    if condition2:
        if condition3:
            if condition4:
                # Deep nesting!
                do_something()
```

✅ **Better (use elif and early returns):**
```python
if not condition1:
    return

if not condition2:
    return

if not condition3:
    return

if condition4:
    do_something()
```

---

## 🎭 Conditional Expressions (Ternary Operator)

### Syntax

```python
value_if_true if condition else value_if_false
```

**Purpose:** Concise way to assign values based on a condition.

### Example 15: Simple Ternary

```python
age = 20

# Traditional if-else
if age >= 18:
    status = "Adult"
else:
    status = "Minor"

# Ternary (one line)
status = "Adult" if age >= 18 else "Minor"

print(status)  # Adult
```

### Example 16: Ternary in Print

```python
score = 85
print("Pass" if score >= 60 else "Fail")  # Pass

temperature = 28
message = "Hot" if temperature > 30 else "Pleasant"
print(message)  # Pleasant
```

### Example 17: Nested Ternary (Use Sparingly)

```python
score = 75

# Nested ternary (hard to read!)
grade = "A" if score >= 90 else "B" if score >= 80 else "C" if score >= 70 else "F"
print(grade)  # C

# Better: Use regular if-elif-else for multiple conditions
```

---

## 📊 Data Science Applications

### Application 1: Data Filtering Based on Conditions

```python
# Filter data points above threshold
data_points = [45, 67, 23, 89, 12, 95, 34]
threshold = 50

filtered_data = []
for value in data_points:
    if value > threshold:
        filtered_data.append(value)

print(f"Original data: {data_points}")
print(f"Filtered (> {threshold}): {filtered_data}")
# Filtered (> 50): [67, 89, 95]
```

### Application 2: Missing Value Handling

```python
# Handle missing values (represented as -999)
raw_score = -999

if raw_score == -999:
    # Impute with mean or remove
    processed_score = None
    print("Missing value detected - marked for imputation")
else:
    processed_score = raw_score
    print(f"Valid score: {processed_score}")
```

### Application 3: Outlier Detection

```python
# Simple outlier detection using IQR method
value = 150
mean = 100
std_dev = 20

# Z-score method: value is outlier if |z| > 3
z_score = abs(value - mean) / std_dev

if z_score > 3:
    print(f"Outlier detected! Z-score: {z_score:.2f}")
    outlier_status = "Yes"
else:
    print(f"Normal value. Z-score: {z_score:.2f}")
    outlier_status = "No"
```

### Application 4: Feature Engineering - Binning

```python
# Create age groups (binning continuous variable)
age = 35

if age < 18:
    age_group = "Child"
elif age < 30:
    age_group = "Young Adult"
elif age < 50:
    age_group = "Middle Age"
elif age < 70:
    age_group = "Senior"
else:
    age_group = "Elderly"

print(f"Age {age} → Category: {age_group}")
# Age 35 → Category: Middle Age
```

### Application 5: Data Validation

```python
# Validate survey response
def validate_survey_response(response):
    """
    Validate survey response is within acceptable range.
    Valid responses: 1-5 (Likert scale)
    """
    if not isinstance(response, int):
        return False, "Error: Response must be an integer"
    
    if response < 1 or response > 5:
        return False, "Error: Response must be between 1 and 5"
    
    return True, "Valid response"

# Test validation
test_responses = [3, 0, 7, "invalid", 5]

for resp in test_responses:
    is_valid, message = validate_survey_response(resp)
    print(f"Response {resp}: {message}")
```

### Application 6: Classification Logic

```python
# Simple rule-based classification
def classify_customer(age, income, credit_score):
    """
    Classify customer for loan approval.
    """
    if age < 18:
        return "Rejected", "Too young"
    
    if credit_score < 600:
        return "Rejected", "Low credit score"
    
    if income < 30000:
        return "Rejected", "Insufficient income"
    
    if credit_score >= 750 and income >= 50000:
        return "Approved - Premium", "Excellent profile"
    
    if credit_score >= 650:
        return "Approved - Standard", "Good profile"
    
    return "Under Review", "Manual review required"

# Test classification
status, reason = classify_customer(30, 60000, 780)
print(f"Status: {status}")
print(f"Reason: {reason}")
```

---

## ⚠️ Common Pitfalls

### Pitfall 1: Using = Instead of ==

❌ **Wrong:**
```python
x = 10
if x = 5:  # SyntaxError
    print("x is 5")
```

✅ **Correct:**
```python
x = 10
if x == 5:  # Use == for comparison
    print("x is 5")
```

### Pitfall 2: Missing Colon

❌ **Wrong:**
```python
if x > 5  # SyntaxError: missing colon
    print("x is greater than 5")
```

✅ **Correct:**
```python
if x > 5:  # Colon is required
    print("x is greater than 5")
```

### Pitfall 3: Incorrect Indentation

❌ **Wrong:**
```python
if x > 5:
print("x is greater than 5")  # IndentationError
```

✅ **Correct:**
```python
if x > 5:
    print("x is greater than 5")  # Proper 4-space indent
```

### Pitfall 4: Comparing Floats for Equality

❌ **Problematic:**
```python
result = 0.1 + 0.2
if result == 0.3:  # False! (floating point precision)
    print("Equal")
```

✅ **Better:**
```python
result = 0.1 + 0.2
tolerance = 1e-9
if abs(result - 0.3) < tolerance:  # Use tolerance for floats
    print("Equal enough")
```

### Pitfall 5: Truthy/Falsy Confusion

❌ **Unexpected behavior:**
```python
user_input = input("Enter name: ")  # User presses Enter (empty string)

if user_input:  # Empty string is Falsy!
    print(f"Hello, {user_input}")
else:
    print("No name entered")  # This executes
```

✅ **Explicit check:**
```python
if user_input != "":  # Explicit comparison
    print(f"Hello, {user_input}")
```

### Pitfall 6: Redundant elif After Comprehensive if

❌ **Redundant:**
```python
if x > 0:
    print("Positive")
elif x <= 0:  # Redundant - else would work
    print("Non-positive")
```

✅ **Cleaner:**
```python
if x > 0:
    print("Positive")
else:  # Simpler
    print("Non-positive")
```

---

## 💻 Python Implementation Examples

### Example 18: Login System

```python
# Simple username/password checker
correct_username = "admin"
correct_password = "python123"

username = input("Enter username: ")
password = input("Enter password: ")

if username == correct_username and password == correct_password:
    print("✓ Login successful!")
    print("Welcome to the system.")
elif username == correct_username:
    print("✗ Incorrect password")
elif password == correct_password:
    print("✗ Incorrect username")
else:
    print("✗ Both username and password are incorrect")
```

### Example 19: BMI Calculator with Categories

```python
# BMI Calculator with health categories
weight = float(input("Enter weight (kg): "))
height = float(input("Enter height (m): "))

bmi = weight / (height ** 2)

print(f"\nYour BMI: {bmi:.2f}")

# Categorize
if bmi < 18.5:
    category = "Underweight"
    advice = "Consider consulting a nutritionist"
elif bmi < 25:
    category = "Normal weight"
    advice = "Keep up the healthy lifestyle!"
elif bmi < 30:
    category = "Overweight"
    advice = "Consider exercise and balanced diet"
else:
    category = "Obese"
    advice = "Recommend consulting a healthcare provider"

print(f"Category: {category}")
print(f"Advice: {advice}")
```

### Example 20: Leap Year Checker

```python
# Determine if a year is a leap year
# Rules:
# 1. Divisible by 4 → leap year
# 2. EXCEPT if divisible by 100 → not leap year
# 3. EXCEPT if divisible by 400 → leap year

year = int(input("Enter a year: "))

if year % 400 == 0:
    is_leap = True
elif year % 100 == 0:
    is_leap = False
elif year % 4 == 0:
    is_leap = True
else:
    is_leap = False

if is_leap:
    print(f"{year} is a leap year")
else:
    print(f"{year} is not a leap year")

# Alternative concise version
is_leap = (year % 400 == 0) or (year % 4 == 0 and year % 100 != 0)
```

### Example 21: Discount Calculator

```python
# Calculate discount based on purchase amount
purchase_amount = float(input("Enter purchase amount: $"))

if purchase_amount >= 1000:
    discount_rate = 0.20  # 20%
    tier = "Platinum"
elif purchase_amount >= 500:
    discount_rate = 0.15  # 15%
    tier = "Gold"
elif purchase_amount >= 200:
    discount_rate = 0.10  # 10%
    tier = "Silver"
else:
    discount_rate = 0.05  # 5%
    tier = "Bronze"

discount_amount = purchase_amount * discount_rate
final_amount = purchase_amount - discount_amount

print(f"\n--- Purchase Summary ---")
print(f"Tier: {tier}")
print(f"Original Amount: ${purchase_amount:.2f}")
print(f"Discount ({discount_rate*100:.0f}%): ${discount_amount:.2f}")
print(f"Final Amount: ${final_amount:.2f}")
print(f"You saved: ${discount_amount:.2f}")
```

---

## 📝 Practice Problems

### Basic Level

**Problem 1:** Write a program to check if a number is positive, negative, or zero.

**Problem 2:** Create a program that determines if a person can vote based on age (18+).

**Problem 3:** Write a program to find the maximum of three numbers.

### Intermediate Level

**Problem 4:** Create a program that categorizes temperature:
- Below 0°C: "Freezing"
- 0-10°C: "Cold"
- 10-25°C: "Pleasant"
- Above 25°C: "Hot"

**Problem 5:** Write a program to determine if a given year is a leap year.

**Problem 6:** Create a simple calculator that takes two numbers and an operator (+, -, *, /) and performs the operation.

### Advanced Level

**Problem 7:** Build a grading system that:
- Takes a numerical score (0-100)
- Assigns letter grade (A-F)
- Adds plus/minus modifiers (e.g., A+, B-, C+)

**Problem 8:** Create a program that validates a password based on rules:
- At least 8 characters
- Contains at least one uppercase letter
- Contains at least one digit
- Contains at least one special character

**Problem 9:** Write a program to classify a triangle based on angles:
- Acute (all angles < 90°)
- Right (one angle = 90°)
- Obtuse (one angle > 90°)

---

## 📚 Additional Resources

### Documentation
- [Python Control Flow](https://docs.python.org/3/tutorial/controlflow.html)
- [Truth Value Testing](https://docs.python.org/3/library/stdtypes.html#truth-value-testing)
- [Boolean Operations](https://docs.python.org/3/library/stdtypes.html#boolean-operations-and-or-not)

### Practice Platforms
- [HackerRank - Conditionals](https://www.hackerrank.com/domains/python?filters%5Bsubdomains%5D%5B%5D=py-if-else)
- [Codewars](https://www.codewars.com/)

### Videos
- [Python Conditionals - Corey Schafer](https://www.youtube.com/watch?v=DZwmZ8Usvnk)

---

**Next Week:** [Week 3: Iterations and Loops](week-03-iterations-loops.md)

---

**Key Takeaways:**
1. ✅ Boolean expressions evaluate to True or False
2. ✅ Use == for comparison, = for assignment
3. ✅ if-elif-else chains execute only ONE block (first True condition)
4. ✅ Logical operators: and (all), or (any), not (invert)
5. ✅ Ternary operator for concise conditional assignment
6. ✅ Avoid deep nesting - use early returns or elif chains
7. ✅ Be careful with truthy/falsy values (0, "", None are Falsy)

**Master conditionals and your programs will start to think! 🧠**
