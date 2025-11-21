# IIT Madras BS Data Science - Project Context

**Last Updated**: November 15, 2025
**Student**: Godson Kurishinkal
**Program**: BS in Data Science and Applications
**Current Level**: Foundation (Term 1)

---

## 🎯 Project Purpose

This repository serves as a **comprehensive learning workspace** for the IIT Madras BS in Data Science and Applications program. It's designed to:

1. **Organize** all study materials systematically across 4+ years
2. **Support AI-assisted learning** with RAG-optimized structure
3. **Track progress** from Foundation → Diploma → BSc → BS → PG Diploma → MTech
4. **Build portfolio** of projects and implementations
5. **Maintain continuity** across study sessions with consistent structure

---

## 📊 Current Status

### Overall Progress
- **Program Level**: Foundation Level (Term 1)
- **Credits Completed**: 0/182 total credits
- **Current Focus**: BSMA1001 (Mathematics for Data Science I)
- **Repository Status**: Scaffolding complete, Week 1 exemplar created

### Foundation Level (32 credits - IN PROGRESS)
| Course Code | Course Name | Credits | Status | Progress |
|-------------|-------------|---------|--------|----------|
| BSMA1001 | Mathematics for Data Science I | 4 | 🔄 In Progress | Week 1/12 complete |
| BSMA1002 | Statistics for Data Science I | 4 | 📝 Not Started | 0/12 weeks |
| BSCS1001 | Computational Thinking | 4 | 📝 Not Started | 0/12 weeks |
| BSHS1001 | English I | 4 | 📝 Not Started | 0/12 weeks |
| BSMA1003 | Mathematics for Data Science II | 4 | 📝 Not Started | 0/11 weeks |
| BSMA1004 | Statistics for Data Science II | 4 | 📝 Not Started | 0/12 weeks |
| BSCS1002 | Python Programming | 4 | 📝 Not Started | 0/12 weeks |
| BSHS1002 | English II | 4 | 📝 Not Started | 0/12 weeks |

**Legend**: ✅ Complete | 🔄 In Progress | 📝 Not Started

---

## 📁 Repository Structure

```
iitm-bs-learning/
│
├── PROJECT_CONTEXT.md              # ⭐ THIS FILE - Master reference for AI assistants
├── README.md                       # Main repository overview
├── .gitignore                      # Git ignore rules
│
├── docs/                           # 📚 Documentation
│   ├── SETUP.md                    # Environment setup guide
│   ├── STUDY_GUIDE.md              # Study strategies and workflows
│   ├── CONTENT_STRATEGY.md         # Content creation approach
│   └── templates/                  # Document templates
│       ├── note-template.md
│       ├── notebook-template.ipynb
│       └── README-template.md
│
├── .venv/                          # 🐍 Python virtual environment
│   └── [Python 3.9.6 + 40+ packages]
│
├── activate.sh                     # Quick activation script
├── requirements.txt                # Python dependencies
│
├── 01-Foundation-Level/            # ⭐ CURRENT LEVEL (32 credits)
│   ├── README.md                   # Foundation overview
│   ├── COMPLETION-STATUS.md        # Detailed progress tracker
│   │
│   ├── 01-Mathematics/             # BSMA1001 + BSMA1003
│   │   ├── README.md               # Course guide
│   │   ├── notes/                  # 📝 Theory & concepts (Markdown)
│   │   │   ├── 00-BSMA1001-overview.md
│   │   │   ├── week-01-notes.md    # ✅ Complete (2000+ words)
│   │   │   ├── week-02-notes.md    # Template
│   │   │   └── week-03...12-notes.md
│   │   ├── notebooks/              # 💻 Code & applications (Jupyter)
│   │   │   ├── week-01-practice.ipynb  # ✅ Complete & functional
│   │   │   └── week-02...12-practice.ipynb
│   │   ├── assignments/            # Course assignments
│   │   ├── practice/               # Additional exercises
│   │   └── resources/              # Reference materials
│   │
│   ├── 02-Statistics/              # BSMA1002 + BSMA1004
│   ├── 03-Python-Programming/      # BSCS1002
│   ├── 04-Computational-Thinking/  # BSCS1001
│   └── 05-English/                 # BSHS1001 + BSHS1002
│
├── 02-Diploma-Level/               # 54 credits
├── 03-BSc-Degree-Level/            # 114 total credits
├── 04-BS-Degree-Level/             # 142 total credits
├── 05-PG-Diploma-Level/            # 162 total credits
├── 06-MTech-Level/                 # 182 total credits
│
└── 99-Resources/                   # Shared resources
    ├── cheatsheets/
    ├── datasets/
    └── templates/
```

---

## 🎓 Content Quality Standards

### Week 1 Exemplar (Quality Benchmark)

**BSMA1001 Week 1** sets the standard for all future content:

#### Notes (`week-01-notes.md`):
- ✅ 2000+ words comprehensive coverage
- ✅ Clear section structure (Key Concepts, Definitions, Formulas, Examples)
- ✅ LaTeX-formatted mathematical notation
- ✅ 6 worked examples with step-by-step solutions
- ✅ Data science applications section
- ✅ Practice problems (basic, intermediate, advanced)
- ✅ Cross-references and resources
- ✅ Connection to next week's topics

#### Notebook (`week-01-practice.ipynb`):
- ✅ Fully functional and tested
- ✅ 7 code cells + 6 markdown cells
- ✅ Professional visualizations (9 plots generated)
- ✅ Real-world application (customer segmentation)
- ✅ Interactive demonstrations
- ✅ Verification of mathematical principles
- ✅ Practice problems section
- ✅ Self-assessment checklist

**All subsequent weeks must match or exceed this quality!**

---

## 🛠️ Technology Stack

### Development Environment
- **Python**: 3.9.6
- **Virtual Environment**: `.venv` (40+ packages installed)
- **IDE**: VS Code with Jupyter, Python, Markdown extensions
- **Version Control**: Git + GitHub

### Core Libraries
**Data Science Stack**:
- NumPy 2.0.2, Pandas 2.3.3, SciPy 1.13.1
- Matplotlib 3.9.4, Seaborn 0.13.2, Plotly 6.4.0
- matplotlib-venn 1.1.2, NetworkX (for graphs)

**Jupyter Environment**:
- JupyterLab 4.4.10, Jupyter Notebook 7.4.7
- ipywidgets 8.1.8, nbformat 5.10.4

**Machine Learning** (for later courses):
- scikit-learn 1.6.1, statsmodels 0.14.5

**Mathematics**:
- SymPy 1.14.0 (symbolic mathematics)

**Development Tools**:
- black 25.11.0, pylint 3.3.9, autopep8 2.3.2
- pytest 8.4.2 (testing)

**Web & Data**:
- requests 2.32.5, beautifulsoup4 4.14.2, lxml 6.0.2

---

## 📖 Study System and Conventions

For complete details on:
- File naming conventions
- Notes vs Notebooks separation
- Quality standards and checklists
- Code standards and documentation
- RAG optimization guidelines

**See**: [conventions-and-standards.md](./conventions-and-standards.md) (single source of truth)

For daily/weekly workflows and study strategies:

**See**: [study-guide.md](./study-guide.md)

---

## 📈 Progress Tracking

### Completed ✅
1. **Repository Structure**: All folders created and organized
2. **Automation System**: 124 template files generated
3. **Week 1 Exemplar**: High-quality notes + functional notebook
4. **Virtual Environment**: .venv with 40+ packages installed
5. **Documentation**: README, guides, templates created
6. **Version Control**: Git initialized, pushed to GitHub

### In Progress 🔄
1. **BSMA1001 Week 2-12**: 11 weeks remaining
2. **Study Workflow**: Establishing daily routine

### Pending 📝
1. **Complete BSMA1001**: Weeks 2-12 (11 weeks)
2. **Other Foundation Courses**: 7 courses remaining
3. **Diploma Level**: 9 courses
4. **BSc Level**: 5 courses
5. **BS Level**: 32 courses
6. **PG Diploma**: 5 courses
7. **MTech**: 5 courses + project

---

## 🎯 Content Creation Strategy

### Current Approach: Incremental Learning-Driven

**Process**:
1. Study week-by-week following IIT Madras schedule
2. Create notes during/after lectures
3. Build notebooks while practicing
4. Polish to match Week 1 quality
5. Commit and push weekly

**Benefits**:
- Authentic learning experience
- Personal insights and challenges
- Better retention
- Natural pace

**Timeline**:
- Per course: 12 weeks
- Foundation level: 6-8 months

### Alternative: AI-Assisted Generation

If needed, can use AI to:
- Generate draft content from syllabus
- Create practice problems
- Build Jupyter notebooks
- Then enhance with personal experience

**See**: `docs/CONTENT_STRATEGY.md` for full details

---

## 🔗 Key Resources

### IIT Madras
- **Portal**: https://ds.study.iitm.ac.in/
- **Forum**: https://discourse.onlinedegree.iitm.ac.in/
- **Support**: support@study.iitm.ac.in | 7850999966
- **Program Info**: https://study.iitm.ac.in/ds/

### Repository
- **GitHub**: https://github.com/GodsonKurishinkal/iitm-bs-learning
- **Latest Commit**: 1b3e1e1 (matplotlib-venn fix)
- **Branch**: main

### External Learning
- Khan Academy (Math/Stats)
- Real Python (Python tutorials)
- StatQuest (Statistics videos)
- Stack Overflow (coding issues)

---

## 📞 Quick Commands

### Environment
```bash
# Activate environment
source activate.sh

# Start Jupyter
jupyter lab

# Install package
pip install package-name

# Deactivate
deactivate
```

### Git
```bash
# Status
git status

# Add and commit
git add -A
git commit -m "message"

# Push
git push

# Pull latest
git pull
```

---

## 💡 Notes for Future Reference

### Lessons Learned
1. **Quality over quantity**: Week 1 exemplar took time but sets standard
2. **Structure matters**: Consistent organization helps AI assistants
3. **Test everything**: All code must be functional before committing
4. **Document continuously**: Don't wait to update documentation

### Automation Insights
- Template generation saved 100+ hours
- JSON data structure enables programmatic content creation
- Week 1 quality can be AI-replicated with proper prompts

### Best Practices Established
- Separate notes (theory) from notebooks (practice)
- Use LaTeX for math notation
- Include visualizations liberally
- Add real-world examples to every topic
- Commit after each week's completion

---

## 🎓 Program Roadmap

**Foundation** (32cr) → Foundation Certificate
↓
**Diploma** (54cr) → Diploma in Programming / Data Science
↓
**BSc** (28cr) → BSc in Programming & Data Science
↓
**BS** (28cr) → BS in Data Science & Applications
↓
**PG Diploma** (20cr) → PG Diploma in AI & ML (CGPA ≥ 8.0 required)
↓
**MTech** (20cr) → MTech in AI & ML

**Total Journey**: 182 credits over 4-8 years

---

**This context file is the master reference for AI assistants helping with this repository. It should be consulted before any major content creation or organizational decisions.**

**Last Updated**: November 15, 2025
**Next Review**: After completing BSMA1001 (12 weeks)
