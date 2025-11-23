# GitHub Copilot Instructions for IIT Madras BS Learning Repository

**Version**: 1.1.0
**Last Updated**: 2025-11-22
**Next Review**: After Foundation Level completion (2026-02)

---

## 🎯 Repository Identity

This is a **RAG-optimized learning workspace** for the IIT Madras BS in Data Science and Applications program. It's designed as an AI-assisted study environment spanning **182 credits over 4-8 years** (Foundation → Diploma → BSc → BS → PG Diploma → MTech).

**Owner**: Godson Kurishinkal
**Repository**: https://github.com/GodsonKurishinkal/iitm-bs-learning
**Purpose**: Personal learning workspace, portfolio building, and AI-assisted study sessions

---

## 📊 Current Status (CRITICAL CONTEXT)

**Program Level**: Foundation Level (32 credits)
**Current Course**: BSMA1001 - Mathematics for Data Science I
**Current Week**: Week 1 (of 12) - ✅ COMPLETE
**Overall Progress**: 0/32 Foundation credits | 0/182 total credits
**Last Updated**: 2025-11-21

**Active Focus**:
- Creating notes and notebooks week-by-week
- Establishing study workflow and habits
- Building quality standard based on Week 1 exemplar

---

## ⭐ CRITICAL QUALITY MANDATE

**Week 1 Exemplar = Quality Standard for ALL Content**

Located in: `01-Foundation-Level/01-Mathematics/`

### Notes Quality Standard (`notes/week-01-*.md`)
✅ **2000+ words** comprehensive coverage
✅ Clear hierarchical structure (H1 → H2 → H3 → H4)
✅ **LaTeX-formatted** mathematical notation (`$...$` inline, `$$...$$` blocks)
✅ **6+ worked examples** with step-by-step solutions
✅ **Data science applications** section (real-world connections)
✅ **Practice problems** at 3 levels (basic, intermediate, advanced)
✅ Cross-references to related topics
✅ Visual aids where applicable (Venn diagrams, function graphs)
✅ Theorems with proofs
✅ Common pitfalls and misconceptions section

### Notebook Quality Standard (`notebooks/week-01-*.ipynb`)
✅ **Fully functional and tested** code (all cells must run)
✅ **7+ code cells + 6+ markdown cells** minimum
✅ **Professional visualizations** (matplotlib, seaborn, plotly)
✅ **9+ plots/visualizations** demonstrating concepts
✅ Real-world applications (e.g., customer segmentation)
✅ Interactive demonstrations of mathematical principles
✅ Practice problems section with solutions
✅ Self-assessment checklist
✅ Well-commented code with docstrings

**ALL subsequent content must match or exceed this quality!**

---

## 📁 File Organization Rules (STRICTLY ENFORCE)

### Theory vs Practice Separation (CRITICAL)

**📝 Notes (Theory)** = Markdown files in `notes/`
- **Purpose**: Study content, theory, concepts, formulas, proofs
- **Format**: Markdown (`.md`)
- **Location**: `[Level]/[Subject]/notes/`
- **Contains**: Definitions, theorems, explanations, worked examples

**💻 Notebooks (Practice)** = Jupyter files in `notebooks/`
- **Purpose**: Hands-on implementation, visualizations, experiments
- **Format**: Jupyter Notebook (`.ipynb`)
- **Location**: `[Level]/[Subject]/notebooks/`
- **Contains**: Code, plots, interactive demos, computational exercises

### File Naming Convention (STRICTLY ENFORCE)

**ALL Markdown Files: Use lowercase-with-hyphens** (no uppercase, no underscores, no spaces)

**Weekly Notes**:
- Format: `week-{nn}-{topic-description}.md`
- ✅ Good: `week-01-set-theory-relations-functions.md`
- ✅ Good: `week-02-coordinate-systems-2d.md`
- ❌ Bad: `Week-01-Notes.md`, `week_01.md`, `BSMA1001-week-01.md`

**Study Guides & Special Files**:
- Format: `study-guide.md`, `completion-status.md`, `notebook-template-guide.md`
- Overview files: `00-[course-code]-overview.md` (e.g., `00-bsma1001-overview.md`)
- ✅ Good: `study-guide.md`, `completion-status.md`, `00-bsma1001-overview.md`
- ❌ Bad: `STUDY-GUIDE.md`, `COMPLETION-STATUS.md`, `00-BSMA1001-overview.md`

**Notebooks**:
- Format: `week-{nn}-{topic}-practice.ipynb`
- ✅ Good: `week-01-set-theory-practice.ipynb`
- ❌ Bad: `Week01.ipynb`, `practice_notebook.ipynb`

**Exception**: `README.md` files remain uppercase (standard convention)

### Course Folder Structure (Standard)
```
[Subject]/
├── README.md                    # Course overview
├── study-guide.md              # 12-week detailed plan
├── notes/                       # 📝 Theory (Markdown)
│   ├── 00-[course-code]-overview.md
│   ├── week-01-topic.md
│   └── week-02-topic.md
├── notebooks/                   # 💻 Practice (Jupyter)
│   ├── week-01-practice.ipynb
│   └── week-02-practice.ipynb
├── assignments/                 # Course submissions
├── practice/                    # Extra exercises
├── projects/                    # Applied projects
└── resources/                   # Reference materials
```

### Metadata Block (Required in All Files)

**For Notes**:
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

**For Notebooks**: Include as first markdown cell

---

## 🛠️ Technology Stack

**Python Environment**:
- Python: 3.9.6
- Virtual Environment: `.venv/` (40+ packages installed)
- Activation: `source activate.sh` or `source .venv/bin/activate`

**Core Libraries** (from `requirements.txt`):
- **Data Science**: NumPy 2.0.2, Pandas 2.3.3, SciPy 1.13.1
- **Visualization**: Matplotlib 3.9.4, Seaborn 0.13.2, Plotly 6.4.0, matplotlib-venn 1.1.2
- **Jupyter**: JupyterLab 4.4.10, Jupyter Notebook 7.4.7, ipywidgets 8.1.8
- **ML**: scikit-learn 1.6.1, statsmodels 0.14.5
- **Math**: SymPy 1.14.0 (symbolic mathematics), NetworkX (graphs)
- **Dev Tools**: black 25.11.0, pylint 3.3.9, autopep8 2.3.2, pytest 8.4.2

**IDE**: VS Code with Python, Jupyter, Markdown, Copilot extensions

---

## 📝 Content Creation Guidelines

### When Creating Notes (Markdown)

1. **Structure**: Use Week 1 exemplar as template
2. **Length**: Aim for 2000+ words comprehensive coverage
3. **Math Notation**: Use LaTeX formatting
   - Inline: `$f(x) = x^2$`
   - Block: `$$\int_{a}^{b} f(x)dx$$`
4. **Worked Examples**: Include 6+ step-by-step examples
5. **Real-World Connections**: Always include "Data Science Applications" section
6. **Practice Problems**: 3 levels (basic, intermediate, advanced)
7. **Cross-References**: Link to related topics with relative paths
8. **Common Pitfalls**: Include misconceptions and how to avoid them

### When Creating Notebooks (Jupyter)

1. **Test Everything**: All code must run without errors
2. **Cell Balance**: Minimum 7 code cells + 6 markdown cells
3. **Visualizations**: Create 9+ professional plots
4. **Interactivity**: Use widgets where appropriate (`ipywidgets`)
5. **Documentation**: Include docstrings and inline comments
6. **Real Applications**: Connect to practical data science problems
7. **Structure**: Introduction → Concepts → Examples → Practice → Assessment
8. **Self-Assessment**: End with checklist of learning objectives

### Code Standards (Python)

- **Style**: PEP 8 compliant (use `black` formatter)
- **Type Hints**: Use where appropriate
- **Docstrings**: Google or NumPy style
- **Testing**: Verify outputs before committing
- **Comments**: Explain complex logic
- **Imports**: Organized (standard → third-party → local)

### Mathematical Content

- **Theorems**: State clearly with proofs
- **Definitions**: Precise mathematical definitions
- **Formulas**: LaTeX formatted with explanations
- **Notation**: Consistent throughout repository
- **Examples**: Show both symbolic and numerical solutions

---

## 🗂️ Key Reference Files (CHECK THESE FIRST)

**Master Context Files**:
1. `docs/project-context.md` ⭐ - Complete repository context, progress, structure
2. `docs/conventions-and-standards.md` ⭐⭐ - File naming, folder structure, quality standards (SINGLE SOURCE OF TRUTH)
3. `docs/ai-assistant-guide.md` - Consolidated guide specifically for AI assistants with quick reference tables

**Progress Tracking**:
1. `01-Foundation-Level/completion-status.md` - Foundation progress tracker
2. `[Level]/[Subject]/README.md` - Course-specific progress

**Templates**:
1. `docs/note-template.md` - Comprehensive RAG-optimized note template (consolidated)
2. `docs/templates/readme-template.md` - Course README structure

**Study Guides**:
1. `docs/study-guide.md` - Complete study system and workflows
2. `docs/quick-start-guide.md` - Environment setup and getting started (consolidated)

**Program Structure**:
1. `docs/program-overview.md` - Complete program path (Foundation → MTech)
2. `docs/content-strategy.md` - Content creation approaches

---

## 🧠 RAG Optimization Principles

This repository is designed for Retrieval-Augmented Generation (RAG) systems.
See `docs/note-template.md` for complete details.

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

## 🎯 AI Assistant Workflows

### Before Creating Content

1. **Check Current Status**: Read `docs/project-context.md` for latest progress
2. **Verify Quality Standard**: Review Week 1 exemplar files
3. **Check Templates**: Use appropriate template from `docs/templates/`
4. **Understand Context**: Read course `README.md` and `study-guide.md` (both lowercase)

### When Creating Weekly Notes

1. Use `docs/templates/note-template.md` as base structure
2. Match Week 1 exemplar quality (2000+ words)
3. Include metadata block at top
4. Use LaTeX for all mathematical notation
5. Add 6+ worked examples
6. Include Data Science Applications section
7. Create 3-level practice problems
8. Cross-reference related topics
9. Save as: `notes/week-{nn}-{topic-description}.md`

### When Creating Notebooks

1. Start with markdown cell explaining purpose
2. Import required libraries (NumPy, Pandas, Matplotlib, etc.)
3. Create 7+ code cells demonstrating concepts
4. Generate 9+ visualizations
5. Include real-world application example
6. Add practice problems section
7. End with self-assessment checklist
8. Test all cells run successfully
9. Save as: `notebooks/week-{nn}-{topic}-practice.ipynb`

### When Suggesting Code

1. **Test First**: Ensure code runs without errors
2. **Follow PEP 8**: Use proper Python style
3. **Add Comments**: Explain logic clearly
4. **Include Docstrings**: Document functions
5. **Use Type Hints**: Where appropriate
6. **Show Output**: Include expected results
7. **Handle Errors**: Add try-except where needed

---

## 🚫 Common Mistakes to Avoid

1. **NEVER** save course notes in `00-RAG-Studies/notes/` - that's for templates only
2. **NEVER** use uppercase or underscores in markdown filenames except `README.md` (use lowercase-with-hyphens)
3. **NEVER** mix theory and practice in same file (separate notes/ and notebooks/)
4. **NEVER** skip metadata blocks in files
5. **NEVER** use uppercase in filenames: `STUDY-GUIDE.md` ❌ → `study-guide.md` ✅
6. **NEVER** use uppercase in course codes: `00-BSMA1001-overview.md` ❌ → `00-bsma1001-overview.md` ✅
7. **NEVER** create generic names (`notes.md`, `practice.ipynb` ❌)
8. **NEVER** commit untested notebook code
9. **NEVER** skip the Data Science Applications section in notes

---

## 📊 Progress Tracking

After creating content:
1. Update `[Level]/completion-status.md`
2. Update course `README.md` progress tracker
3. Mark week as complete in `study-guide.md`
4. Update `docs/project-context.md` if milestone reached

---

## 🔄 Update Schedule

**Review and update this file**:
- After each Foundation course completion (~12 weeks)
- When transitioning between levels (Foundation → Diploma, etc.)
- When quality standards evolve
- When new workflows are established
- Major milestone: After Foundation Level completion (2026-02)

---

## 💡 Quick Reference

| Task | Check This File |
|------|----------------|
| Current progress | `docs/project-context.md` |
| File naming rules | `docs/conventions-and-standards.md` (or this file) |
| Note template | `docs/note-template.md` |
| Quality standard | Week 1 exemplar in `01-Foundation-Level/01-Mathematics-I/` |
| Study workflow | `docs/study-guide.md` |
| Environment setup | `docs/quick-start-guide.md` |
| Course structure | Subject `README.md` and `study-guide.md` files |
| Progress tracking | `[Level]/completion-status.md` |

---

## 🎓 Learning Philosophy

**Quality Over Quantity**: Match Week 1 exemplar quality always
**Theory + Practice**: Always provide both notes and notebooks
**Real-World Focus**: Every concept connects to data science
**Test Everything**: All code must be functional before committing
**Consistent Structure**: Follow templates and conventions strictly
**RAG-Optimized**: Make content easily retrievable for AI assistants

---

**Remember**: This repository will be used for 4-8 years spanning Foundation through MTech. Maintain high quality and consistency from the start!

**Version History**:
- v1.1.0 (2025-11-22): Standardized ALL markdown filenames to lowercase-with-hyphens
- v1.0.0 (2025-11-21): Initial Copilot instructions created
