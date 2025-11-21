# Quick Start Guide

**Version**: 1.0.0
**Last Updated**: 2025-11-21
**For**: Foundation Level (getting started)

> **Purpose**: Get up and running fast. Complete environment setup, understand the workspace structure, and start your first week of learning.

---

## 🚀 Fast Track Setup (5 Minutes)

### Step 1: Activate Python Environment

**Option A: Using activation script** (Recommended)
```bash
cd /Users/godsonkurishinkal/Projects/iitm-bs-learning
source activate.sh
```

**Option B: Manual activation**
```bash
source .venv/bin/activate
```

You should see `(.venv)` in your terminal prompt.

### Step 2: Verify Installation
```bash
python --version          # Should show Python 3.9.6
pip list                  # 40+ packages installed
jupyter --version         # Verify Jupyter works
```

### Step 3: Start Jupyter
```bash
jupyter lab               # Modern interface (recommended)
# OR
jupyter notebook          # Classic interface
```

### Step 4: Open First Notebook
- Navigate to: `01-Foundation-Level/01-Mathematics-I/notebooks/`
- Open: `week-01-set-theory-relations-functions.ipynb`
- Run all cells to verify everything works

**✅ You're ready to learn!**

---

## 📁 Understanding Your Workspace

### Six-Level Program Structure

```
01-Foundation-Level/     ← YOU ARE HERE (Week 1/12)
02-Diploma-Level/        ← Future
03-BSc-Degree-Level/     ← Future
04-BS-Degree-Level/      ← Future
05-PG-Diploma-Level/     ← Future
06-MTech-Level/          ← Future
```

### Current Focus: Foundation Level

**8 Courses | 32 Credits | ~8 months**

```
01-Foundation-Level/
├── 01-Mathematics-I/         ← Current course
│   ├── notes/                📝 Theory (Markdown)
│   ├── notebooks/            💻 Practice (Jupyter)
│   ├── assignments/          Submissions
│   └── resources/            Reference materials
├── 01-Mathematics-II/
├── 02-Statistics-I/
├── 02-Statistics-II/
├── 03-Python-Programming/
├── 04-Computational-Thinking/
└── 05-English/
```

---

## 📝 Key Principle: Notes vs Notebooks

### 📝 Notes = Theory

**Location**: `[Subject]/notes/`
**Format**: Markdown (`.md`)
**Purpose**: Study concepts, understand theory

**Contains**:
- Definitions and theorems
- Formulas and equations (LaTeX)
- Proofs and derivations
- Conceptual explanations
- Worked examples

**When to Use**: After lectures, while reading textbook, when building understanding

### 💻 Notebooks = Practice

**Location**: `[Subject]/notebooks/`
**Format**: Jupyter (`.ipynb`)
**Purpose**: Implement concepts, experiment, visualize

**Contains**:
- Code implementations
- Visualizations and plots
- Computational examples
- Interactive demonstrations
- Practice exercises

**When to Use**: After understanding theory, for hands-on practice, when building portfolio

**Why Separation?**
- Clarity of purpose
- Better organization
- Easier to find content
- Optimized for AI retrieval

---

## 🎯 First Week Action Plan

### Day 1 (Today)
- [x] ✅ Environment setup (you just did this!)
- [ ] Read this entire quick start guide
- [ ] Open Week 1 exemplar notebook and run all cells
- [ ] Read [conventions-and-standards.md](./conventions-and-standards.md)

### Day 2-3
- [ ] Attend first lectures
- [ ] Create your first note: `notes/week-01-{topic}.md`
- [ ] Use metadata template from conventions
- [ ] Try Python basics notebook

### Day 4-5
- [ ] Complete practice exercises in notebooks
- [ ] Create visualizations for concepts learned
- [ ] Start first assignment
- [ ] Review course README for resources

### Day 6-7
- [ ] Review week's notes (theory)
- [ ] Run through week's notebooks (practice)
- [ ] Complete all practice problems
- [ ] Update progress tracker in course README
- [ ] Plan next week

---

## 📚 Essential Documentation

Read these in order:

1. **This file** - Getting started ⭐ (you are here)
2. **[conventions-and-standards.md](./conventions-and-standards.md)** - File naming, quality standards
3. **[study-guide.md](./study-guide.md)** - Daily/weekly workflows
4. **[project-context.md](./project-context.md)** - Current status, progress tracking
5. **Subject README** - Course-specific guidance

---

## 🛠️ Environment Details

### What's Installed

**Core Stack** (Python 3.9.6):
- **NumPy 2.0.2** - Numerical computing
- **Pandas 2.3.3** - Data manipulation
- **SciPy 1.13.1** - Scientific computing
- **SymPy 1.14.0** - Symbolic mathematics

**Visualization**:
- **Matplotlib 3.9.4** - Static plots
- **Seaborn 0.13.2** - Statistical viz
- **Plotly 6.4.0** - Interactive plots
- **matplotlib-venn 1.1.2** - Venn diagrams

**Jupyter**:
- **JupyterLab 4.4.10** - Modern interface
- **Jupyter Notebook 7.4.7** - Classic interface
- **ipywidgets 8.1.8** - Interactive widgets

**Machine Learning**:
- **scikit-learn 1.6.1** - ML algorithms
- **statsmodels 0.14.5** - Statistical models

**Development**:
- **black 25.11.0** - Code formatter
- **pylint 3.3.9** - Linter
- **pytest 8.4.2** - Testing

Total: **40+ packages**

### VS Code Setup

**Recommended Extensions** (see `.vscode/extensions.json`):
- Python (ms-python.python)
- Jupyter (ms-toolsai.jupyter)
- GitHub Copilot (github.copilot)
- Markdown All in One (yzhang.markdown-all-in-one)
- GitLens (eamodio.gitlens)

**Select Python Interpreter**:
1. Cmd+Shift+P (Mac) / Ctrl+Shift+P (Windows)
2. Type: "Python: Select Interpreter"
3. Choose: `.venv/bin/python`

---

## ❓ Troubleshooting

### Environment Won't Activate
```bash
# Recreate from scratch
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Package Import Error
```bash
# Make sure environment is activated
source .venv/bin/activate

# Verify package installed
pip list | grep numpy

# Reinstall if needed
pip install --upgrade numpy
```

### Jupyter Kernel Not Found
```bash
pip install ipykernel
python -m ipykernel install --user --name=iitm-bs
```

Then in Jupyter: Kernel → Change Kernel → iitm-bs

### VS Code Not Finding Modules
1. Check Python interpreter is set to `.venv/bin/python`
2. Reload VS Code window (Cmd+Shift+P → "Reload Window")
3. Restart Jupyter kernel in notebook

### Git Issues
```bash
# Check status
git status

# If files not tracked
git add .
git commit -m "Your message"
git push
```

---

## 💻 Common Commands

### Virtual Environment
```bash
# Activate
source .venv/bin/activate

# Deactivate
deactivate

# Check what's installed
pip list

# Install new package
pip install package-name

# Update requirements.txt
pip freeze > requirements.txt
```

### Jupyter
```bash
# Start JupyterLab
jupyter lab

# Start classic Notebook
jupyter notebook

# List running servers
jupyter server list

# Stop all servers
jupyter server stop
```

### Git
```bash
# Check status
git status

# Add changes
git add .

# Commit with message
git commit -m "Week 1 complete"

# Push to GitHub
git push

# Pull latest changes
git pull
```

---

## 📋 File Naming Quick Reference

### Weekly Content
```
✅ Good:
notes/week-01-set-theory-relations-functions.md
notebooks/week-01-set-theory-practice.ipynb
notes/week-02-coordinate-systems-2d.md

❌ Bad:
notes/Week-01-Notes.md           (uppercase, not descriptive)
notebooks/practice.ipynb          (too generic)
notes/BSMA1001-week-01.md        (course code included)
```

### General Rules
- Use **lowercase-with-hyphens**
- No underscores (`_`), no spaces, no CamelCase
- Include week number: `week-01`, `week-02` (2 digits)
- Be descriptive: 2-4 word topic
- NO course codes in weekly files

---

## 🎓 Learning Workflow

### Daily Routine (1-2 hours)

**Theory Phase** (30-60 mins):
1. Watch lecture or read chapter
2. Take notes in `notes/week-XX-{topic}.md`
3. Include metadata block (see conventions)
4. Use LaTeX for math: `$f(x) = x^2$`
5. Add worked examples

**Practice Phase** (30-60 mins):
1. Open `notebooks/week-XX-practice.ipynb`
2. Implement concepts in code
3. Create visualizations
4. Experiment with variations
5. Save and run all cells

**Review** (10 mins):
- Update progress tracker
- Note questions
- Plan tomorrow

### Weekly Routine

**Mon-Wed**: Theory Focus
- Attend lectures
- Create/update notes
- Read textbook chapters

**Thu-Fri**: Practice Focus
- Work through notebooks
- Code implementations
- Create visualizations

**Sat**: Application
- Complete assignments
- Work on projects
- Real-world applications

**Sun**: Review & Integration
- Review all notes
- Run all notebooks
- Connect theory ↔ practice
- Self-assessment
- Plan next week

---

## 🎯 Quality Standards (The Week 1 Exemplar)

### Your Target Quality

**Location**: `01-Foundation-Level/01-Mathematics-I/`

**Notes** (`notes/week-01-*.md`):
- ✅ 2000+ words
- ✅ LaTeX math notation
- ✅ 6+ worked examples
- ✅ Data Science Applications section
- ✅ 3-level practice problems
- ✅ Cross-references

**Notebooks** (`notebooks/week-01-*.ipynb`):
- ✅ All cells run successfully
- ✅ 7+ code cells, 6+ markdown cells
- ✅ 9+ professional plots
- ✅ Real-world application
- ✅ Well-commented code
- ✅ Self-assessment checklist

**Match or exceed this quality for ALL future content!**

---

## 📞 Getting Help

### IIT Madras Support
- **Portal**: https://ds.study.iitm.ac.in/
- **Forum**: https://discourse.onlinedegree.iitm.ac.in/
- **Email**: support@study.iitm.ac.in
- **Phone**: 7850999966

### Online Resources
- **Stack Overflow** - Coding questions
- **Khan Academy** - Math/Stats basics
- **Real Python** - Python tutorials
- **StatQuest** - Statistics videos (YouTube)

### Within This Repository
- Read documentation in `docs/`
- Check course `README.md` files
- Review exemplar Week 1 content
- Check `.github/copilot-instructions.md` for Copilot context

---

## ✅ Success Checklist

### Environment Setup
- [ ] Python environment activates
- [ ] Jupyter Lab/Notebook starts
- [ ] Can run Week 1 exemplar notebook
- [ ] VS Code extensions installed
- [ ] Git configured

### Knowledge
- [ ] Understand notes vs notebooks separation
- [ ] Know file naming conventions
- [ ] Familiar with folder structure
- [ ] Read conventions-and-standards.md
- [ ] Know quality standards (Week 1 exemplar)

### First Week
- [ ] Created first note with metadata
- [ ] Completed first notebook exercise
- [ ] Updated progress tracker
- [ ] Established daily routine
- [ ] Planned next week

---

## 🎉 You're Ready!

Your workspace is professionally organized and ready for 4-8 years of learning:
- ✅ Environment configured
- ✅ Tools installed
- ✅ Structure established
- ✅ Quality standards clear
- ✅ Workflow defined

**Next Steps**:
1. Read [conventions-and-standards.md](./conventions-and-standards.md) (15 mins)
2. Review [study-guide.md](./study-guide.md) (20 mins)
3. Explore Week 1 exemplar content (30 mins)
4. Start your first lecture and create notes!

---

**Remember**:
> "The secret of getting ahead is getting started." - Mark Twain

You've organized yourself professionally. Now maintain this quality throughout your journey! 🚀

**Good luck with your IIT Madras BS in Data Science! You've got this! 💪**
