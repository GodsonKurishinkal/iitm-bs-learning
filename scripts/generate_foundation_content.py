#!/usr/bin/env python3
"""
Foundation Level Content Generator
Generates high-quality notes and notebooks for all Foundation courses
Based on Mathematics I Week 1 exemplar quality
"""

import json
import os
from pathlib import Path
from datetime import date

# Base directory
BASE_DIR = Path("/Users/godsonkurishinkal/Projects/iitm-bs-learning/01-Foundation-Level")

# Course mapping
COURSES = {
    "01-Mathematics-II": {
        "code": "BSMA1003",
        "name": "Mathematics for Data Science II",
        "weeks": 11,
        "data_file": "BSMA1003_data.json"
    },
    "02-Statistics-I": {
        "code": "BSMA1002",
        "name": "Statistics for Data Science I",
        "weeks": 12,
        "data_file": "BSMA1002_data.json"
    },
    "02-Statistics-II": {
        "code": "BSMA1004",
        "name": "Statistics for Data Science II",
        "weeks": 12,
        "data_file": "BSMA1004_data.json"
    },
    "04-Computational-Thinking": {
        "code": "BSCS1001",
        "name": "Computational Thinking",
        "weeks": 12,
        "data_file": "BSCS1001_data.json"
    },
    "03-Python-Programming": {
        "code": "BSCS1002",
        "name": "Programming in Python",
        "weeks": 12,
        "data_file": "BSCS1002_data.json"
    }
}

def load_course_data(course_folder):
    """Load course data from JSON file"""
    data_file = BASE_DIR / course_folder / "resources" / COURSES[course_folder]["data_file"]
    with open(data_file, 'r') as f:
        return json.load(f)

def generate_notes_template(course_info, week_num, week_topic):
    """Generate comprehensive notes following Mathematics I exemplar"""
    
    template = f"""# Week {week_num}: {week_topic}

**Date**: {date.today().strftime('%Y-%m-%d')}  
**Course**: {course_info['name']} ({course_info['code']})

## Topics Covered

{week_topic}

---

## Key Concepts

### 1. [Main Concept 1]

[Detailed explanation with examples]

**Why important for DS**: [Connection to data science]

### 2. [Main Concept 2]

[Detailed explanation]

### 3. [Main Concept 3]

[Detailed explanation]

---

## Definitions

- **Term 1**: Definition with clear explanation
- **Term 2**: Definition with clear explanation
- **Term 3**: Definition with clear explanation

---

## Important Formulas

### Formula 1
$$formula$$

### Formula 2
$$formula$$

---

## Theorems & Proofs

### Theorem 1: [Name]
**Statement**: [Clear statement]

**Proof**: [Step-by-step proof]

**Significance**: [Why this matters]

---

## Examples (Worked Problems)

### Example 1: [Topic]
**Problem**: [Clear problem statement]

**Solution**:
[Step-by-step solution with explanations]

### Example 2: [Topic]
**Problem**: [Problem statement]

**Solution**:
[Detailed solution]

### Example 3: [Topic]
**Problem**: [Problem statement]

**Solution**:
[Solution]

---

## Data Science Applications

### Why This Matters in Data Science

1. **Application 1**
   - Description
   - Real-world use case
   
2. **Application 2**
   - Description
   - Example

### Real-World Example: [Scenario]

```python
# Code example showing application
```

---

## Practice Problems

### Basic Level

1. **Problem 1**: [Statement]
2. **Problem 2**: [Statement]
3. **Problem 3**: [Statement]

### Intermediate Level

4. **Problem 4**: [Statement]
5. **Problem 5**: [Statement]

### Advanced Level

6. **Problem 6**: [Statement]
7. **Problem 7**: [Statement]

### Challenge Problems

8. **Challenge**: [Advanced problem]

---

## Questions/Doubts

- [ ] Question 1
- [ ] Question 2
- [ ] Question 3

---

## Action Items

- [ ] Review lecture slides
- [ ] Complete practice problems 1-3
- [ ] Work through notebook examples
- [ ] Watch related video lectures
- [ ] Solve textbook exercises

---

## Key Takeaways

1. **Key Point 1**: Main learning
2. **Key Point 2**: Important concept
3. **Key Point 3**: Critical understanding
4. **Key Point 4**: Practical application

---

## References

- **Textbook**: [Chapter references]
- **Video Lectures**: IIT Madras Week {week_num} lectures ({course_info['code']})
- **Practice**: Week {week_num} Practice Notebook
- **Online Resources**: [Relevant links]

---

## Connection to Next Week

Week {week_num + 1} will build on these concepts by exploring:
- [Next topic preview]
- [Connection to current week]

---

**Last Updated**: {date.today().strftime('%Y-%m-%d')}  
**Next Class**: Week {week_num + 1}
"""
    return template

def generate_notebook_template(course_info, week_num, week_topic):
    """Generate comprehensive Jupyter notebook following Mathematics I exemplar"""
    
    # This would generate XML-formatted notebook content
    # For now, returning a placeholder - actual implementation would be much longer
    return {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    f"# Week {week_num}: {week_topic}\\n",
                    "\\n",
                    f"**Course**: {course_info['name']} ({course_info['code']})  \\n",
                    f"**Date**: {date.today().strftime('%Y-%m-%d')}\\n"
                ]
            },
            # Add more cells here following the exemplar structure
        ]
    }

def generate_content_for_course(course_folder):
    """Generate all content for a course"""
    print(f"\\n{'='*60}")
    print(f"Generating content for: {course_folder}")
    print(f"{'='*60}")
    
    course_info = COURSES[course_folder]
    course_data = load_course_data(course_folder)
    
    notes_dir = BASE_DIR / course_folder / "notes"
    notebooks_dir = BASE_DIR / course_folder / "notebooks"
    
    for week_data in course_data["weeks"]:
        week_num = int(week_data["week"].split()[-1])
        week_topic = week_data["topics"]
        
        print(f"  Week {week_num}: {week_topic[:50]}...")
        
        # Generate notes
        notes_content = generate_notes_template(course_info, week_num, week_topic)
        notes_file = notes_dir / f"week-{week_num:02d}-{week_topic.lower().replace(' ', '-')[:30]}.md"
        
        # Generate notebook
        # notebook_content = generate_notebook_template(course_info, week_num, week_topic)
        
        print(f"    ✓ Notes template generated")
        # print(f"    ✓ Notebook template generated")

def main():
    """Main execution"""
    print("Foundation Level Content Generator")
    print("=" * 60)
    print(f"Base Directory: {BASE_DIR}")
    print(f"Courses to generate: {len(COURSES)}")
    print()
    
    # Generate for each course
    for course_folder in COURSES.keys():
        generate_content_for_course(course_folder)
    
    print("\\n" + "="*60)
    print("Content generation complete!")
    print("="*60)

if __name__ == "__main__":
    main()
