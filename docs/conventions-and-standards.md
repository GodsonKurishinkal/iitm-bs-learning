# Repository Conventions and Standards

**Version**: 2.0.0
**Last Updated**: 2025-11-21
**Next Review**: After Foundation Level completion (2026-02)

> **Purpose**: This is the single source of truth for all repository conventions, quality standards, and coding practices. All other documentation references this file.

---

## 📁 1. Folder Structure Conventions

### Six-Level Program Hierarchy

```
01-Foundation-Level/     (32 credits, 8 courses)
02-Diploma-Level/        (54 credits, 2 diplomas)
03-BSc-Degree-Level/     (28 credits)
04-BS-Degree-Level/      (28 credits)
05-PG-Diploma-Level/     (20 credits, CGPA ≥ 8.0 required)
06-MTech-Level/          (20 credits, research thesis)
```

### Standard Course Folder Structure

Every course follows this identical organization:

```
[Course-Name]/
├── README.md           # Course overview, weekly topics, resources
├── notes/              # 📝 THEORY: Markdown study content
├── notebooks/          # 💻 PRACTICE: Jupyter implementations
├── assignments/        # Course submissions
├── practice/           # Extra exercises
├── projects/           # Applied projects (where applicable)
└── resources/          # PDFs, links, formula sheets
```

---

## 📝 2. File Naming Conventions

### Weekly Content Files

**Notes (Markdown)**:
- Format: `week-{nn}-{topic-description}.md`
- Use **lowercase with hyphens** (no underscores, no spaces, no CamelCase)
- Include 2-4 word descriptive topic
- Use two-digit week numbers (`01`, `02`, not `1`, `2`)
- NO course codes in filenames

**✅ Good Examples**:
```
week-01-set-theory-relations-functions.md
week-02-coordinate-systems-2d.md
week-03-straight-lines-slopes.md
week-10-graph-theory-basics.md
```

**❌ Bad Examples**:
```
Week-01-Notes.md              (uppercase, not descriptive)
week_01.md                    (underscores, not descriptive)
BSMA1001-week-01.md          (course code included)
notes.md                      (too generic)
SetTheory.md                  (CamelCase, no week number)
```

**Notebooks (Jupyter)**:
- Format: `week-{nn}-{topic}-practice.ipynb`

**✅ Good**: `week-01-set-theory-practice.ipynb`
**❌ Bad**: `Week01.ipynb`, `practice_notebook.ipynb`

### Special Files

- Course overview: `00-[COURSE-CODE]-overview.md` (only place to use course code)
- Course README: `README.md` (always uppercase)
- Documentation: `lowercase-with-hyphens.md` (e.g., `study-guide.md`)

---

## 🗂️ 3. Theory vs Practice Separation (CRITICAL)

### 📝 Notes = Theory (Markdown)

**Location**: `[Subject]/notes/`
**Format**: Markdown (`.md`)
**Purpose**: Study content, theoretical understanding

**Contains**:
- Lecture summaries and key concepts
- Definitions and theorems
- Formulas and equations (LaTeX formatted)
- Proofs and derivations
- Conceptual explanations
- Worked examples (step-by-step solutions)

**Philosophy**: Master the "what" and "why"

### 💻 Notebooks = Practice (Jupyter)

**Location**: `[Subject]/notebooks/`
**Format**: Jupyter Notebook (`.ipynb`)
**Purpose**: Hands-on implementation, experiments

**Contains**:
- Code examples demonstrating concepts
- Worked solutions with code
- Visualizations and plots
- Computational experiments
- Practice exercises with implementations
- Real-world applications
- Interactive demonstrations

**Philosophy**: Apply the "how" through coding

### Why Separation Matters

1. **Clarity**: Theory and practice serve different purposes
2. **Workflow**: Read notes first, then practice in notebooks
3. **Organization**: Easy to find what you need
4. **RAG Optimization**: AI can retrieve appropriate content type
5. **Portfolio**: Notebooks showcase coding skills

---

## 📋 4. Metadata Requirements

### For Markdown Notes

Place at the very top of the file:

```markdown
---
Date: YYYY-MM-DD
Course: [CODE] - [Full Name]
Level: Foundation | Diploma | BSc | BS | PG Diploma | MTech
Week: [N] of [Total]
Source: IIT Madras [Course] Week [N]
Topic Area: Mathematics | Statistics | Programming | ML | etc.
Tags: #CourseCode #Topic #WeekN #Level
---
```

**Example**:
```markdown
---
Date: 2025-11-15
Course: BSMA1001 - Mathematics for Data Science I
Level: Foundation
Week: 1 of 12
Source: IIT Madras Mathematics I Week 1
Topic Area: Mathematics
Tags: #BSMA1001 #SetTheory #Week1 #Foundation
---
```

### For Jupyter Notebooks

Include as the **first markdown cell**:

```markdown
# Week [N]: [Topic Name]

**Course**: [CODE] - [Full Name]
**Date**: YYYY-MM-DD
**Level**: Foundation | Diploma | BSc | BS
**Source**: IIT Madras [Course] Week [N]

---

## Learning Objectives

[List objectives here]
```

---

## ⭐ 5. Quality Standards (Week 1 Exemplar)

**Location**: `01-Foundation-Level/01-Mathematics/`

### Notes Quality Checklist

✅ **2000+ words** comprehensive coverage
✅ Clear hierarchical structure (H1 → H2 → H3 → H4)
✅ **LaTeX-formatted** mathematical notation
   - Inline: `$f(x) = x^2$`
   - Block: `$$\int_{a}^{b} f(x)dx$$`
✅ **6+ worked examples** with step-by-step solutions
✅ **Data Science Applications** section (real-world connections)
✅ **Practice problems** at 3 levels:
   - Basic (understanding check)
   - Intermediate (application)
   - Advanced (synthesis/challenge)
✅ Cross-references to related topics
✅ Visual aids where applicable (diagrams, graphs)
✅ Theorems stated with proofs
✅ Common pitfalls and misconceptions section
✅ Self-assessment checklist at end

### Notebook Quality Checklist

✅ **Fully functional and tested** (all cells run without errors)
✅ **7+ code cells + 6+ markdown cells** minimum
✅ **Professional visualizations**:
   - matplotlib (static plots)
   - seaborn (statistical)
   - plotly (interactive)
✅ **9+ plots/visualizations** demonstrating concepts
✅ Real-world applications (e.g., data analysis, segmentation)
✅ Interactive demonstrations using ipywidgets where appropriate
✅ Practice problems section with solutions
✅ Self-assessment checklist
✅ Well-commented code with docstrings
✅ Type hints for functions
✅ Clear markdown explanations between code sections

### Content Structure Standards

**For Notes**:
1. Metadata block
2. Overview and learning objectives
3. Key concepts (definitions, theorems)
4. Detailed explanations
5. Worked examples (6+)
6. Data science applications
7. Practice problems (3 levels)
8. Self-assessment
9. Resources and next steps

**For Notebooks**:
1. Metadata markdown cell
2. Introduction and objectives
3. Import libraries
4. Conceptual demonstrations (code + viz)
5. Real-world application example
6. Practice problems
7. Self-assessment checklist

---

## 💻 6. Code Standards

### Python Style (PEP 8)

**Formatting**:
- Use `black` formatter (line length 88)
- Organize imports: standard → third-party → local
- 2 blank lines between top-level functions/classes
- 1 blank line between methods

**Naming**:
- `snake_case` for functions and variables
- `PascalCase` for classes
- `UPPER_CASE` for constants
- Descriptive names (no `x`, `y`, `z` except in math contexts)

### Documentation Standards

**Docstrings** (Google style):
```python
def calculate_mean(values: list[float]) -> float:
    """
    Calculate the arithmetic mean of a list of values.

    Args:
        values: List of numeric values

    Returns:
        The arithmetic mean as a float

    Raises:
        ValueError: If values list is empty

    Example:
        >>> calculate_mean([1, 2, 3, 4, 5])
        3.0
    """
    if not values:
        raise ValueError("Cannot calculate mean of empty list")
    return sum(values) / len(values)
```

**Type Hints**:
```python
from typing import List, Tuple, Optional

def process_data(
    data: List[float],
    threshold: float = 0.5
) -> Tuple[List[float], List[float]]:
    """Process data and return above/below threshold."""
    above = [x for x in data if x > threshold]
    below = [x for x in data if x <= threshold]
    return above, below
```

### Visualization Standards

**Required Elements**:
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for consistency
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# Create plot with labels
plt.plot(x, y)
plt.title("Clear Descriptive Title", fontsize=14, fontweight='bold')
plt.xlabel("X-axis Label with Units", fontsize=12)
plt.ylabel("Y-axis Label with Units", fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

**Best Practices**:
- Always include title, axis labels, legend
- Use colorblind-friendly palettes
- Add grid for readability
- Save plots with `plt.savefig()` for documentation
- Use `tight_layout()` to prevent label cutoff

---

## 📐 7. Mathematical Notation Standards

### LaTeX Formatting

**Inline Math** (within text):
```markdown
The function $f(x) = x^2$ is a parabola.
```

**Block Math** (standalone equations):
```markdown
$$
\int_{a}^{b} f(x)dx = F(b) - F(a)
$$
```

### Common Patterns

**Sets**:
```markdown
$A = \{1, 2, 3\}$
$A \cup B$ (union)
$A \cap B$ (intersection)
$A \subseteq B$ (subset)
```

**Functions**:
```markdown
$f: X \rightarrow Y$
$f(x) = ax + b$
```

**Calculus**:
```markdown
$\frac{d}{dx}f(x)$ (derivative)
$\int f(x)dx$ (integral)
$\lim_{x \to 0} f(x)$ (limit)
```

**Linear Algebra**:
```markdown
$\vec{v} = \begin{bmatrix} x \\ y \\ z \end{bmatrix}$
$A = \begin{bmatrix} a & b \\ c & d \end{bmatrix}$
```

---

## 🧠 8. RAG Optimization Guidelines

### Hierarchical Information Architecture
- **H1**: Main topic (big picture)
- **H2**: Major sections (theory, implementation, application)
- **H3**: Specific concepts
- **H4**: Implementation details

### Front-Loading Critical Information
- Key insights within first 100 words
- Formulas and definitions near beginning
- Summary bullet points immediately accessible

### Self-Contained Sections
- Each H2 section can stand alone
- Includes context within section
- Strategic redundancy for retrieval accuracy

### Semantic Markers (Use in Content)
```markdown
**Definition:** Clear statement
**Purpose:** Why this exists
**Use Case:** When to apply
**Pros/Cons:** Trade-offs
**Complexity:** O(n) analysis
**Alternatives:** Other methods
```

### Consistent Terminology
- Use glossary terms consistently
- Document aliases: "Feature (also: predictor, independent variable, attribute)"

---

## 🔗 9. Cross-Referencing Standards

### Internal Links (Relative Paths)

**Link to other notes**:
```markdown
See also: [Coordinate Systems](./week-02-coordinate-systems-2d.md)
```

**Link to notebooks**:
```markdown
Try the practice: [Set Theory Notebook](../notebooks/week-01-set-theory-practice.ipynb)
```

**Link to resources**:
```markdown
Reference: [Formula Sheet](../resources/formula-sheet.md)
```

### Linking Patterns

**Previous/Next Week**:
```markdown
**Previous**: [Week 0 - Overview](./week-00-overview.md)
**Next**: [Week 2 - Coordinate Systems](./week-02-coordinate-systems-2d.md)
```

**Related Topics**:
```markdown
**Related Concepts**:
- [Functions and Relations](./week-01-set-theory-relations-functions.md#functions)
- [Set Operations](./week-01-set-theory-relations-functions.md#set-operations)
```

---

## 🚫 10. Common Mistakes to Avoid

### File Organization
1. ❌ Mixing theory and practice in same file
2. ❌ Using uppercase in filenames (except README)
3. ❌ Using underscores instead of hyphens
4. ❌ Including course codes in weekly filenames
5. ❌ Generic filenames (`notes.md`, `practice.ipynb`)

### Content Quality
1. ❌ Skipping metadata blocks
2. ❌ Notes shorter than 1500 words (aim for 2000+)
3. ❌ Notebooks with untested code
4. ❌ Missing Data Science Applications section
5. ❌ No practice problems or only one difficulty level
6. ❌ Missing cross-references to related topics
7. ❌ Plots without titles/labels/legends

### Code Quality
1. ❌ Code without docstrings
2. ❌ No type hints
3. ❌ Poor variable names (`x`, `temp`, `data1`)
4. ❌ No error handling
5. ❌ Inconsistent formatting (not using black)

---

## 📊 11. Progress Tracking Standards

### Course-Level Tracking

Each course `README.md` includes:

```markdown
| Week | Topic | Status | Notes |
|------|-------|--------|-------|
| 1 | Set Theory | ✅ Complete | Exemplar quality |
| 2 | Coordinate Systems | 🔄 In Progress | 60% done |
| 3 | Straight Lines | 📝 Not Started | |
```

**Status Indicators**:
- ✅ Complete (note + notebook + tested)
- 🔄 In Progress (actively working)
- 📝 Not Started (planned)

---

## 📅 12. Update and Maintenance

### When to Update This Document

- New course level started (Diploma, BSc, etc.)
- Quality standards evolve
- New tools or libraries adopted
- File structure changes
- New conventions established

### Version History

- **v2.0.0** (2025-11-21): Consolidated from multiple files, added all standards
- **v1.0.0** (2025-11-21): Initial conventions document created

---

## 🔗 Related Documentation

- **[project-context.md](./project-context.md)**: Current status and progress
- **[study-guide.md](./study-guide.md)**: Daily/weekly workflows
- **[note-template.md](./note-template.md)**: Complete note template
- **[quick-start-guide.md](./quick-start-guide.md)**: Environment setup

---

**Remember**: This is the single source of truth for all conventions. Follow these standards strictly to maintain consistency across your 4-8 year learning journey!
