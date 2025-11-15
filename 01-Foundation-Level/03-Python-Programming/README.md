# Python Programming - IIT Madras Foundation

## 📖 Course Overview

Foundation level Python programming covering basics to intermediate concepts essential for Data Science.

## 📚 Topics Covered

### Module 1: Python Basics (Weeks 1-3)
- Variables and data types
- Operators
- Input/Output
- Conditional statements
- Loops (for, while)

### Module 2: Data Structures (Weeks 4-6)
- Lists
- Tuples
- Sets
- Dictionaries
- Strings (advanced)

### Module 3: Functions (Weeks 7-8)
- Function definition
- Parameters and arguments
- Return values
- Scope
- Lambda functions

### Module 4: File Handling (Week 9)
- Reading files
- Writing files
- CSV handling
- JSON handling

### Module 5: Object-Oriented Programming (Weeks 10-11)
- Classes and objects
- Inheritance
- Encapsulation
- Polymorphism

### Module 6: Libraries & Modules (Week 12)
- NumPy basics
- Pandas introduction
- Matplotlib basics
- Working with modules

## 📂 Folder Contents

### `/notes`
Lecture notes and summaries:
- `week-01-basics.md`
- `week-02-conditionals-loops.md`
- `cheatsheet.md`

### `/notebooks`
Jupyter notebooks for hands-on practice:
- `01-basics-practice.ipynb` - Starter template ✅
- `02-data-structures.ipynb`
- `03-functions.ipynb`
- `04-oop.ipynb`

### `/assignments`
Course assignments:
```
assignments/
├── assignment-01-basics/
│   ├── problem.md
│   └── solution.py
├── assignment-02-lists/
└── ...
```

### `/practice`
Additional coding exercises:
- `coding-exercises.md`
- Daily practice problems
- Challenge problems

### `/projects`
Mini-projects:
- `calculator/`
- `todo-app/`
- `data-analyzer/`

### `/resources`
- Python cheatsheets
- Useful links
- Code snippets library

## 🎯 Learning Tips

1. **Code daily** - Even 30 minutes makes a difference
2. **Type, don't copy** - Build muscle memory
3. **Debug yourself** - Learn to read error messages
4. **Start projects** - Apply what you learn
5. **Read others' code** - Learn different approaches

## 💻 Setup

### Python Installation
```bash
# Check Python version
python3 --version

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux

# Install essential packages
pip install numpy pandas matplotlib jupyter
```

### VS Code Extensions
- Python (Microsoft)
- Jupyter
- Pylance
- Python Indent

## 🔧 Essential Libraries

```python
# Data manipulation
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# File handling
import json
import csv
```

## 📝 Practice Routine

### Daily (30 mins)
- Solve 2-3 coding problems
- Review one concept
- Write clean code

### Weekly
- Complete one Jupyter notebook
- Build a small project
- Review previous week's code

## 🔗 Resources

### Online Platforms
- [Python.org Documentation](https://docs.python.org/3/)
- [Real Python Tutorials](https://realpython.com/)
- [Codecademy Python](https://www.codecademy.com/learn/learn-python-3)
- [LeetCode Easy Problems](https://leetcode.com/problemset/)

### Books
- *Python Crash Course* by Eric Matthes
- *Automate the Boring Stuff* by Al Sweigart

### Video Tutorials
- Corey Schafer's Python Tutorials
- Sentdex Python Programming

## ✅ Progress Tracker

### Concepts
- [ ] Variables & Data Types
- [ ] Conditionals & Loops
- [ ] Lists & Tuples
- [ ] Dictionaries & Sets
- [ ] Functions
- [ ] File Handling
- [ ] OOP Basics
- [ ] NumPy & Pandas

### Projects
- [ ] Calculator
- [ ] Number Guessing Game
- [ ] To-Do List
- [ ] Data Analyzer
- [ ] Web Scraper (if covered)

## 🐛 Common Errors & Solutions

### IndentationError
```python
# Wrong
if x > 5:
print(x)

# Correct
if x > 5:
    print(x)
```

### IndexError
```python
# Check list length before accessing
if len(my_list) > index:
    value = my_list[index]
```

### KeyError (Dictionaries)
```python
# Use .get() method
value = my_dict.get('key', 'default_value')
```

---

**Last Updated**: November 14, 2025
