# Project Structure Reference

> Extended context for GitHub Copilot - IIT Madras BS Data Science

## Complete Directory Structure

```
iitm-bs-learning/
├── .github/
│   ├── copilot-instructions.md          # Core Copilot instructions
│   └── instructions/                     # Path-specific instructions
│       ├── notebooks.instructions.md     # Jupyter notebook patterns
│       └── python.instructions.md        # Python coding standards
├── .vscode/
│   ├── settings.json                     # Workspace settings
│   └── extensions.json                   # Recommended extensions
├── docs/
│   ├── IITM-BS-Course-Catalog.md        # Full course catalog
│   └── copilot-context/                  # Extended context files
│       ├── project-structure.md          # This file
│       ├── python-patterns.md            # Data science patterns
│       └── course-conventions.md         # Course naming conventions
├── 01-Foundation-Level/
│   ├── 01-Mathematics-I/
│   │   ├── notebooks/
│   │   │   └── week-XX-topic.ipynb
│   │   └── resources/
│   │       └── BSMA1001_data.json
│   ├── 01-Mathematics-II/
│   ├── 02-Statistics-I/
│   ├── 02-Statistics-II/
│   ├── 03-Python-Programming/
│   ├── 04-Computational-Thinking/
│   └── 05-English/
├── 02-Diploma-Level/
│   ├── 01-Programming/
│   └── 02-Data-Science/
├── 03-BSc-Degree-Level/
├── 04-BS-Degree-Level/
├── 05-PG-Diploma-Level/
├── 06-MTech-Level/
│   ├── notebooks/
│   ├── project/
│   ├── research/
│   └── resources/
├── requirements.txt
└── README.md
```

## Course Code Pattern

| Prefix | Meaning |
|--------|---------|
| `BSMA` | Mathematics |
| `BSCS` | Computer Science |
| `BSDA` | Data Analytics |
| `BSMS` | Management Studies |
| `BSEE` | Electrical Engineering |
| `BSBT` | Biotechnology |
| `BSHS` | Humanities |
| `BSGN` | General |
| `BSSE` | Systems Engineering |

## Level Identification (4th digit)

| Digit | Level |
|-------|-------|
| 1 | Foundation Level |
| 2 | Diploma Level |
| 3 | BSc Degree Level |
| 4 | BS Degree Level |
| 5 | PG Diploma Level |
| 6 | MTech Level |

## Practical Course Indicator
- Course codes ending with `P` indicate practical/lab courses
- Example: `BSCS2003P` = Modern Application Development I - Project

## Resource Files

### JSON Course Data Structure
```json
{
  "course_code": "BSMA1001",
  "course_name": "Mathematics for Data Science I",
  "credits": 4,
  "prerequisites": [],
  "weeks": [
    {
      "week": 1,
      "topic": "Set Theory",
      "subtopics": ["Number system", "Sets", "Relations"]
    }
  ]
}
```

## Notebook Naming Convention

Format: `week-XX-topic-name.ipynb`

Examples:
- `week-01-introduction-to-sets.ipynb`
- `week-02-coordinate-systems-and-straight-lines.ipynb`
- `week-03-quadratic-functions.ipynb`

## Data File Convention

Format: `data_description_YYYYMMDD.csv`

Examples:
- `demand_forecast_20241126.csv`
- `inventory_levels_20241101.csv`
- `sales_timeseries_20241015.csv`
