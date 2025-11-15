# Python Basics Cheatsheet

## Data Types

```python
# Numbers
x = 42              # int
y = 3.14            # float
z = 2 + 3j          # complex

# Strings
s = "Hello"
s = 'World'
s = """Multi
line"""

# Boolean
flag = True
flag = False

# Collections
lst = [1, 2, 3]           # list (mutable)
tpl = (1, 2, 3)           # tuple (immutable)
dct = {'a': 1, 'b': 2}    # dict
st = {1, 2, 3}            # set
```

## String Operations

```python
s = "Python"
s[0]              # 'P' - indexing
s[-1]             # 'n' - negative indexing
s[0:3]            # 'Pyt' - slicing
s.upper()         # 'PYTHON'
s.lower()         # 'python'
s.strip()         # remove whitespace
s.split(',')      # split by delimiter
s.replace('P', 'J')  # 'Jython'
len(s)            # 6
```

## List Operations

```python
lst = [1, 2, 3]
lst.append(4)         # [1, 2, 3, 4]
lst.extend([5, 6])    # [1, 2, 3, 4, 5, 6]
lst.insert(0, 0)      # [0, 1, 2, 3, 4, 5, 6]
lst.remove(3)         # [0, 1, 2, 4, 5, 6]
lst.pop()             # 6, lst = [0, 1, 2, 4, 5]
lst.sort()            # sort in place
sorted(lst)           # return sorted copy
lst.reverse()         # reverse in place
```

## Dictionary Operations

```python
d = {'a': 1, 'b': 2}
d['c'] = 3            # add/update
d.get('a')            # 1
d.get('z', 0)         # 0 (default)
d.keys()              # dict_keys(['a', 'b', 'c'])
d.values()            # dict_values([1, 2, 3])
d.items()             # dict_items([('a', 1), ...])
'a' in d              # True
del d['b']            # remove key
```

## Control Flow

```python
# If-elif-else
if condition:
    # do something
elif other_condition:
    # do something else
else:
    # default action

# For loop
for item in iterable:
    print(item)

for i in range(10):
    print(i)

# While loop
while condition:
    # do something

# Break and continue
for i in range(10):
    if i == 5:
        break      # exit loop
    if i % 2 == 0:
        continue   # skip to next iteration
```

## List Comprehensions

```python
# Basic
squares = [x**2 for x in range(10)]

# With condition
evens = [x for x in range(10) if x % 2 == 0]

# Nested
matrix = [[i*j for j in range(3)] for i in range(3)]

# Dict comprehension
squares_dict = {x: x**2 for x in range(5)}

# Set comprehension
unique = {x % 3 for x in range(10)}
```

## Functions

```python
# Basic function
def greet(name):
    return f"Hello, {name}!"

# Default arguments
def power(base, exponent=2):
    return base ** exponent

# Variable arguments
def sum_all(*args):
    return sum(args)

# Keyword arguments
def info(**kwargs):
    for key, value in kwargs.items():
        print(f"{key}: {value}")

# Lambda (anonymous function)
square = lambda x: x**2
```

## Common Built-in Functions

```python
len(obj)              # length
type(obj)             # type
str(obj)              # convert to string
int(obj)              # convert to int
float(obj)            # convert to float
range(start, stop, step)  # range object
enumerate(iterable)   # index and value
zip(iter1, iter2)     # combine iterables
map(func, iterable)   # apply function
filter(func, iter)    # filter elements
sum(iterable)         # sum elements
min(iterable)         # minimum
max(iterable)         # maximum
sorted(iterable)      # sorted copy
```

## File I/O

```python
# Read file
with open('file.txt', 'r') as f:
    content = f.read()        # read all
    lines = f.readlines()     # list of lines

# Write file
with open('file.txt', 'w') as f:
    f.write('Hello\n')
    f.writelines(['line1\n', 'line2\n'])

# Append
with open('file.txt', 'a') as f:
    f.write('Appended text\n')
```

## Exception Handling

```python
try:
    # code that might raise exception
    result = 10 / 0
except ZeroDivisionError:
    # handle specific exception
    print("Cannot divide by zero")
except Exception as e:
    # handle any exception
    print(f"Error: {e}")
else:
    # runs if no exception
    print("Success")
finally:
    # always runs
    print("Cleanup")
```

## Common String Methods

```python
s = "  Hello, World!  "
s.strip()             # "Hello, World!"
s.lstrip()            # "Hello, World!  "
s.rstrip()            # "  Hello, World!"
s.startswith('Hello') # False (leading spaces)
s.endswith('!')       # False (trailing spaces)
s.find('World')       # 9
s.count('l')          # 3
s.isalpha()           # False
s.isdigit()           # False
s.isalnum()           # False
```
