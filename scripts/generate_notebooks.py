#!/usr/bin/env python3
"""
Script to generate comprehensive Jupyter notebooks for all Foundation level courses.
Each notebook covers a week's worth of content with theory, code, and practice problems.
"""

import json
import os
from pathlib import Path

# Course structure with all weeks and topics
COURSES = {
    "01-Mathematics": {
        "code": "BSMA1001",
        "name": "Mathematics I",
        "notebooks_dir": "notebooks",
        "weeks": [
            {"week": 1, "topic": "set-theory-relations-functions", "title": "Set Theory, Relations & Functions"},
            {"week": 2, "topic": "coordinate-systems-straight-lines", "title": "Coordinate Systems & Straight Lines"},
            {"week": 3, "topic": "quadratic-functions", "title": "Quadratic Functions & Parabolas"},
            {"week": 4, "topic": "algebra-polynomials", "title": "Algebra & Polynomials"},
            {"week": 5, "topic": "functions-exponential-inverse", "title": "Exponential & Inverse Functions"},
            {"week": 6, "topic": "logarithmic-functions", "title": "Logarithmic Functions"},
            {"week": 7, "topic": "sequences-limits-continuity", "title": "Sequences, Limits & Continuity"},
            {"week": 8, "topic": "derivatives-critical-points", "title": "Derivatives & Critical Points"},
            {"week": 9, "topic": "integrals-areas", "title": "Integrals & Areas"},
            {"week": 10, "topic": "graph-theory-bfs-dfs", "title": "Graph Theory: BFS & DFS"},
            {"week": 11, "topic": "shortest-paths-spanning-trees", "title": "Shortest Paths & Spanning Trees"},
            {"week": 12, "topic": "review-revision", "title": "Review & Revision"},
        ]
    },
    "01-Mathematics-II": {
        "code": "BSMA1003",
        "name": "Mathematics II",
        "notebooks_dir": "notebooks",
        "weeks": [
            {"week": 1, "topic": "vectors-matrices-intro", "title": "Vectors & Matrices Introduction"},
            {"week": 2, "topic": "matrix-operations", "title": "Matrix Operations"},
            {"week": 3, "topic": "linear-equations-systems", "title": "Linear Equations & Systems"},
            {"week": 4, "topic": "determinants", "title": "Determinants"},
            {"week": 5, "topic": "gaussian-elimination", "title": "Gaussian Elimination"},
            {"week": 6, "topic": "vector-spaces", "title": "Vector Spaces"},
            {"week": 7, "topic": "basis-dimension", "title": "Basis & Dimension"},
            {"week": 8, "topic": "rank-nullity", "title": "Rank & Nullity Theorem"},
            {"week": 9, "topic": "optimization-basics", "title": "Optimization Basics"},
            {"week": 10, "topic": "ml-applications", "title": "Machine Learning Applications"},
            {"week": 11, "topic": "advanced-topics", "title": "Advanced Topics"},
        ]
    },
    "02-Statistics": {
        "code": "BSMA1002",
        "name": "Statistics I",
        "notebooks_dir": "notebooks",
        "weeks": [
            {"week": 1, "topic": "data-types-scales", "title": "Data Types & Measurement Scales"},
            {"week": 2, "topic": "categorical-data-analysis", "title": "Categorical Data Analysis"},
            {"week": 3, "topic": "numerical-data-visualization", "title": "Numerical Data Visualization"},
            {"week": 4, "topic": "central-tendency-measures", "title": "Measures of Central Tendency"},
            {"week": 5, "topic": "dispersion-variability", "title": "Dispersion & Variability"},
            {"week": 6, "topic": "correlation-association", "title": "Correlation & Association"},
            {"week": 7, "topic": "probability-basics", "title": "Probability Fundamentals"},
            {"week": 8, "topic": "random-variables", "title": "Random Variables"},
            {"week": 9, "topic": "discrete-distributions", "title": "Discrete Probability Distributions"},
            {"week": 10, "topic": "continuous-distributions", "title": "Continuous Probability Distributions"},
            {"week": 11, "topic": "normal-distribution", "title": "Normal Distribution"},
            {"week": 12, "topic": "applications-review", "title": "Statistical Applications & Review"},
        ]
    },
    "02-Statistics-II": {
        "code": "BSMA1004",
        "name": "Statistics II",
        "notebooks_dir": "notebooks",
        "weeks": [
            {"week": 1, "topic": "multiple-random-variables", "title": "Multiple Random Variables"},
            {"week": 2, "topic": "independence-events", "title": "Independence & Conditional Events"},
            {"week": 3, "topic": "expectations-variance", "title": "Expectations & Variance"},
            {"week": 4, "topic": "continuous-distributions-advanced", "title": "Advanced Continuous Distributions"},
            {"week": 5, "topic": "sampling-methods", "title": "Sampling Methods & Techniques"},
            {"week": 6, "topic": "estimation-theory", "title": "Estimation Theory"},
            {"week": 7, "topic": "hypothesis-testing-intro", "title": "Hypothesis Testing Introduction"},
            {"week": 8, "topic": "hypothesis-testing-advanced", "title": "Advanced Hypothesis Testing"},
            {"week": 9, "topic": "chi-square-tests", "title": "Chi-Square Tests"},
            {"week": 10, "topic": "linear-regression", "title": "Simple Linear Regression"},
            {"week": 11, "topic": "multiple-regression", "title": "Multiple Regression Analysis"},
            {"week": 12, "topic": "applications-review", "title": "Real-World Applications & Review"},
        ]
    },
    "03-Python-Programming": {
        "code": "BSCS1002",
        "name": "Python Programming",
        "notebooks_dir": "notebooks",
        "weeks": [
            {"week": 1, "topic": "algorithms-problem-solving", "title": "Algorithms & Problem Solving"},
            {"week": 2, "topic": "conditionals-control-flow", "title": "Conditionals & Control Flow"},
            {"week": 3, "topic": "iterations-loops", "title": "Iterations & Loops"},
            {"week": 4, "topic": "ranges-sequences", "title": "Ranges & Sequences"},
            {"week": 5, "topic": "lists-tuples", "title": "Lists & Tuples"},
            {"week": 6, "topic": "dictionaries-sets", "title": "Dictionaries & Sets"},
            {"week": 7, "topic": "random-math-libraries", "title": "Random & Math Libraries"},
            {"week": 8, "topic": "datetime-library", "title": "DateTime Library"},
            {"week": 9, "topic": "scipy-numpy", "title": "SciPy & NumPy"},
            {"week": 10, "topic": "matplotlib-visualization", "title": "Matplotlib Visualization"},
            {"week": 11, "topic": "pandas-data-analysis", "title": "Pandas Data Analysis"},
            {"week": 12, "topic": "final-project", "title": "Final Project: Data Analysis Pipeline"},
        ]
    },
    "04-Computational-Thinking": {
        "code": "BSCS1001",
        "name": "Computational Thinking",
        "notebooks_dir": "notebooks",
        "weeks": [
            {"week": 1, "topic": "variables-expressions", "title": "Variables & Expressions"},
            {"week": 2, "topic": "iterators-loops", "title": "Iterators & Loops"},
            {"week": 3, "topic": "filtering-data", "title": "Filtering & Data Selection"},
            {"week": 4, "topic": "data-types-structures", "title": "Data Types & Structures"},
            {"week": 5, "topic": "flowcharts-algorithms", "title": "Flowcharts & Algorithm Design"},
            {"week": 6, "topic": "pseudocode-logic", "title": "Pseudocode & Logic"},
            {"week": 7, "topic": "procedures-functions", "title": "Procedures & Functions"},
            {"week": 8, "topic": "nested-iterations", "title": "Nested Iterations"},
            {"week": 9, "topic": "recursion-basics", "title": "Recursion Basics"},
            {"week": 10, "topic": "advanced-recursion", "title": "Advanced Recursion"},
            {"week": 11, "topic": "debugging-testing", "title": "Debugging & Testing"},
            {"week": 12, "topic": "applications-review", "title": "Applications & Review"},
        ]
    },
    "05-English": {
        "code": "BSHS1001/1002",
        "name": "English",
        "notebooks_dir": "notebooks",
        "weeks": [
            {"week": 1, "topic": "vowel-consonant-sounds", "title": "Vowel & Consonant Sounds"},
            {"week": 2, "topic": "pronunciation-practice", "title": "Pronunciation Practice"},
            {"week": 3, "topic": "intonation-stress", "title": "Intonation & Stress Patterns"},
            {"week": 4, "topic": "sentence-patterns", "title": "Sentence Patterns & Structure"},
            {"week": 5, "topic": "grammar-fundamentals", "title": "Grammar Fundamentals"},
            {"week": 6, "topic": "writing-skills", "title": "Writing Skills Development"},
            {"week": 7, "topic": "reading-comprehension", "title": "Reading Comprehension"},
            {"week": 8, "topic": "listening-skills", "title": "Listening Skills"},
            {"week": 9, "topic": "speaking-practice", "title": "Speaking Practice"},
            {"week": 10, "topic": "academic-writing", "title": "Academic Writing"},
            {"week": 11, "topic": "presentation-skills", "title": "Presentation Skills"},
            {"week": 12, "topic": "review-assessment", "title": "Review & Assessment"},
        ]
    },
}

def create_notebook_structure(course_name, course_code, week_num, topic, title):
    """Create a comprehensive notebook structure"""
    
    notebook = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    # Header cell
    header = f"""# Week {week_num:02d}: {title}

**Course:** {course_name} ({course_code})  
**Duration:** Week {week_num}  
**Last Updated:** November 2025

---

## Learning Objectives
- Understand fundamental concepts and theories
- Apply concepts through practical examples
- Develop problem-solving skills
- Build computational implementations

---"""
    
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": header
    })
    
    # Imports cell
    if "Python" in course_name or "Computational" in course_name or "Mathematics" in course_name or "Statistics" in course_name:
        imports = """# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
%matplotlib inline"""
        
        notebook["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": imports,
            "outputs": []
        })
    
    # Theory section
    theory = f"""## 1. Introduction & Theory

### Overview
This section covers the fundamental concepts of {title.lower()}.

### Key Concepts
- Concept 1: [To be filled with course-specific content]
- Concept 2: [To be filled with course-specific content]
- Concept 3: [To be filled with course-specific content]

### Definitions
Add important definitions and formulas here.

---"""
    
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": theory
    })
    
    # Examples section
    examples = """## 2. Examples & Demonstrations

### Example 1: Basic Concepts
Demonstrate fundamental concepts with clear examples."""
    
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": examples
    })
    
    # Code example
    code_example = """# Example implementation
# Add your code here

print("Example output")"""
    
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": code_example,
        "outputs": []
    })
    
    # Applications section
    applications = """## 3. Practical Applications

### Real-World Use Cases
Explore how these concepts apply in practice.

### Implementation
Step-by-step implementation guide."""
    
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": applications
    })
    
    # Application code
    app_code = """# Practical application
# Implement real-world scenarios here

# Your code here"""
    
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": app_code,
        "outputs": []
    })
    
    # Practice problems
    practice = """## 4. Practice Problems

### Problem 1
**Question:** [Add problem statement]

**Solution Approach:**
1. Step 1
2. Step 2
3. Step 3"""
    
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": practice
    })
    
    # Problem solution code
    solution_code = """# Solution to Problem 1
# Your solution here

# Test your solution"""
    
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": solution_code,
        "outputs": []
    })
    
    # Additional problems
    more_practice = """### Problem 2
**Question:** [Add problem statement]"""
    
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": more_practice
    })
    
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": "# Solution to Problem 2\n# Your solution here",
        "outputs": []
    })
    
    # Exercises section
    exercises = """## 5. Exercises & Challenges

### Exercise 1: Beginner Level
[Add exercise description]

### Exercise 2: Intermediate Level
[Add exercise description]

### Exercise 3: Advanced Level
[Add exercise description]"""
    
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": exercises
    })
    
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": "# Your exercise solutions here",
        "outputs": []
    })
    
    # Summary section
    summary = f"""## Summary

### Key Takeaways
- **Concept 1**: Main learning point
- **Concept 2**: Main learning point
- **Concept 3**: Main learning point

### Important Formulas/Patterns
List key formulas, patterns, or concepts to remember.

### Common Pitfalls
- Pitfall 1: [Description and how to avoid]
- Pitfall 2: [Description and how to avoid]

### Further Reading
- Resource 1: [Link or reference]
- Resource 2: [Link or reference]

---

### Next Week
Week {week_num + 1:02d}: [Next topic]

### Previous Week
"""
    if week_num > 1:
        summary += f"Week {week_num - 1:02d}: [Previous topic]"
    else:
        summary += "This is the first week"
    
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": summary
    })
    
    # Notes section
    notes = """## Personal Notes & Reflections

Use this space for:
- Your own observations
- Questions to explore further
- Connections to other topics
- Important insights

---

**Study Tips:**
1. Review concepts regularly
2. Practice problems multiple times
3. Explain concepts to others
4. Build mini-projects using these concepts"""
    
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": notes
    })
    
    return notebook

def generate_all_notebooks():
    """Generate notebooks for all courses"""
    base_path = Path(__file__).parent.parent / "01-Foundation-Level"
    
    total_notebooks = 0
    
    for course_dir, course_info in COURSES.items():
        course_path = base_path / course_dir
        notebooks_path = course_path / course_info["notebooks_dir"]
        
        # Create notebooks directory if it doesn't exist
        notebooks_path.mkdir(exist_ok=True)
        
        print(f"\nGenerating notebooks for {course_info['name']}...")
        
        for week_info in course_info["weeks"]:
            filename = f"week-{week_info['week']:02d}-{week_info['topic']}.ipynb"
            filepath = notebooks_path / filename
            
            # Skip if notebook already exists
            if filepath.exists():
                print(f"  ✓ {filename} (already exists)")
                continue
            
            notebook = create_notebook_structure(
                course_info['name'],
                course_info['code'],
                week_info['week'],
                week_info['topic'],
                week_info['title']
            )
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(notebook, f, indent=1, ensure_ascii=False)
            
            print(f"  ✓ Created: {filename}")
            total_notebooks += 1
    
    print(f"\n{'='*60}")
    print(f"Total notebooks created: {total_notebooks}")
    print(f"{'='*60}")
    
    return total_notebooks

if __name__ == "__main__":
    print("=" * 60)
    print("Foundation Level Notebook Generator")
    print("=" * 60)
    
    total = generate_all_notebooks()
    
    print("\n✅ Notebook generation complete!")
    print(f"\nAll notebooks have been created in their respective")
    print(f"course 'notebooks/' directories.")
    print(f"\nYou can now:")
    print("1. Open any notebook in Jupyter or VS Code")
    print("2. Fill in course-specific content")
    print("3. Add examples and practice problems")
    print("4. Customize based on your learning needs")
