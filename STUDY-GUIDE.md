# Study Organization Guide

## 📁 Complete Folder Structure

```
Learning/
│
├── README.md                          # Main overview (RAG focus)
├── STUDY-GUIDE.md                     # This file - complete organization guide
│
├── 00-RAG-Studies/                    # RAG learning materials
│   ├── notes/
│   ├── experiments/
│   └── projects/
│
├── 01-Foundation-Level/               # 🎓 Current focus
│   ├── README.md                      # Foundation overview
│   │
│   ├── 01-Mathematics/
│   │   ├── README.md                  # Course guide
│   │   ├── notes/                     # Lecture notes (.md files)
│   │   ├── notebooks/                 # Jupyter practice
│   │   │   └── 01-math-with-python.ipynb ✅
│   │   ├── assignments/               # Course assignments
│   │   ├── practice/                  # Extra problems
│   │   └── resources/                 # Formula sheets, links
│   │
│   ├── 02-Statistics/
│   │   ├── README.md
│   │   ├── notes/
│   │   ├── notebooks/
│   │   │   └── 01-descriptive-stats.ipynb ✅
│   │   ├── assignments/
│   │   ├── practice/
│   │   └── resources/
│   │
│   ├── 03-Python-Programming/
│   │   ├── README.md
│   │   ├── notes/
│   │   ├── notebooks/
│   │   │   └── 01-basics-practice.ipynb ✅
│   │   ├── assignments/
│   │   ├── practice/
│   │   ├── projects/                  # Mini Python projects
│   │   └── resources/
│   │
│   ├── 04-Computational-Thinking/
│   │   ├── notes/
│   │   ├── notebooks/
│   │   ├── assignments/
│   │   └── resources/
│   │
│   └── 05-English/
│       ├── notes/
│       ├── assignments/
│       └── resources/
│
└── 99-Resources/                      # Shared resources
    ├── templates/                     # Note templates
    ├── cheatsheets/                   # Quick references
    └── datasets/                      # Practice datasets
```

## 🎯 How to Use This Structure

### Daily Workflow

#### 1. **During Lectures** (Real-time)
```
01-Foundation-Level/
└── [Subject]/
    └── notes/
        └── week-XX-topic.md
```
- Take notes in Markdown format
- Use template: Date, Topic, Key Points, Questions
- Save as: `week-01-introduction.md`

#### 2. **After Lectures** (Review)
```
01-Foundation-Level/
└── [Subject]/
    └── notebooks/
        └── week-XX-practice.ipynb
```
- Open relevant Jupyter notebook
- Practice coding examples
- Complete exercises
- Document learnings in notebook

#### 3. **Assignments** (Submission)
```
01-Foundation-Level/
└── [Subject]/
    └── assignments/
        └── assignment-XX/
            ├── problem.md
            ├── solution.py (or .ipynb)
            └── submission.pdf
```

#### 4. **Extra Practice** (Skill Building)
```
01-Foundation-Level/
└── [Subject]/
    └── practice/
        └── topic-exercises.md
```

### Weekly Workflow

**Monday - Wednesday**: Lectures + Notes
- Attend live/recorded sessions
- Take notes in `/notes` folder
- Clarify doubts immediately

**Thursday - Friday**: Practice + Code
- Work through Jupyter notebooks
- Complete practice problems
- Apply concepts in code

**Saturday**: Assignments
- Complete course assignments
- Submit on time
- Document solution approach

**Sunday**: Review + Plan
- Review week's material
- Update progress tracker
- Plan next week

## 📝 Note-Taking System

### Markdown Notes Template

Create this in each subject's `/notes` folder:

```markdown
# Topic Name

**Date**: 2025-11-14  
**Week**: 1  
**Subject**: Mathematics  
**Tags**: #algebra #basics #important

## Lecture Summary
Brief overview of what was covered

## Key Concepts

### Concept 1
Explanation with examples

### Concept 2
Explanation with examples

## Important Formulas
- Formula 1: explanation
- Formula 2: explanation

## Examples
### Example 1
Problem statement
Solution steps

## Questions/Doubts
- [ ] Question 1
- [ ] Question 2

## Action Items
- [ ] Practice problems 1-5
- [ ] Review chapter 2
- [ ] Complete assignment

## Links & Resources
- [Resource 1](URL)
- Video: Link

---
**Next Class**: Topic name
```

### Jupyter Notebook Template

Each notebook should have:

1. **Header Cell** (Markdown)
   - Title
   - Date
   - Topics covered

2. **Setup Cell** (Code)
   - Import libraries
   - Load data

3. **Learning Sections** (Markdown + Code)
   - Concept explanation
   - Code examples
   - Practice exercises

4. **Notes Section** (Markdown)
   - What you learned
   - Questions
   - Next steps

## 🗂️ File Naming Conventions

### Notes (Markdown)
```
week-01-introduction.md
week-02-linear-equations.md
2025-11-14-calculus-limits.md
topic-probability-basics.md
```

### Notebooks (Jupyter)
```
01-basics-practice.ipynb
02-data-structures.ipynb
week-03-loops.ipynb
assignment-01-solution.ipynb
```

### Assignments
```
assignment-01/
assignment-02-statistics/
project-calculator/
```

## 📊 Progress Tracking

Create `progress.md` in each subject folder:

```markdown
# [Subject] Progress Tracker

## Week 1
- [x] Attended lecture 1
- [x] Completed notes
- [x] Jupyter notebook practice
- [x] Assignment submitted
- [ ] Extra practice

## Week 2
- [ ] Attended lecture 2
- [ ] Completed notes
...

## Topics Mastered
- [x] Variables and data types
- [x] Conditional statements
- [ ] Loops
- [ ] Functions

## Weak Areas (Review Needed)
- Recursion
- OOP concepts

## Assignments
| Assignment | Due Date | Status | Score |
|------------|----------|--------|-------|
| 1          | Nov 20   | ✅     | 95%   |
| 2          | Nov 27   | 🔄     | -     |
```

## 🎓 Study Strategies

### For Mathematics
1. **Understand, don't memorize** - Focus on concepts
2. **Solve manually first** - Then verify with Python
3. **Visualize** - Use matplotlib to plot functions
4. **Practice daily** - 5-10 problems minimum

### For Statistics
1. **Visualize distributions** - Create plots for everything
2. **Real data practice** - Use actual datasets
3. **Understand formulas** - Know when to use what
4. **Python + Theory** - Learn both simultaneously

### For Python
1. **Code daily** - Even 30 minutes helps
2. **Type, don't copy** - Build muscle memory
3. **Debug yourself** - Learn from errors
4. **Build projects** - Apply what you learn

### For All Subjects
- **Space repetition** - Review material regularly
- **Active recall** - Test yourself without notes
- **Teach others** - Explain concepts to solidify understanding
- **Connect concepts** - See relationships between subjects

## 🔧 Tools Setup

### Essential Tools
```bash
# Python environment
python3 -m venv venv
source venv/bin/activate

# Install packages
pip install numpy pandas matplotlib seaborn scipy jupyter

# Start Jupyter
jupyter notebook
```

### VS Code Extensions
- Python
- Jupyter
- Markdown All in One
- Markdown Preview Enhanced
- Todo Tree

## 📅 Semester Planning

### Beginning of Semester
1. ✅ Create folder structure
2. ✅ Set up Jupyter notebooks
3. ✅ Install required packages
4. Create progress trackers
5. Set study schedule

### During Semester
1. Take notes consistently
2. Practice in Jupyter notebooks
3. Complete assignments on time
4. Review weekly
5. Ask questions promptly

### Before Exams
1. Review all notes
2. Redo practice problems
3. Create summary sheets
4. Mock tests
5. Focus on weak areas

## 🆘 When You're Stuck

### General Approach
1. **Read error message** - It usually tells you the problem
2. **Google it** - Someone has faced this before
3. **Check documentation** - Official docs are gold
4. **Ask in forums** - IIT Madras discourse
5. **Office hours** - Talk to TAs/professors

### Debugging Checklist
- [ ] Read the error message carefully
- [ ] Check for typos
- [ ] Verify data types
- [ ] Print intermediate values
- [ ] Simplify the problem
- [ ] Start fresh if needed

## 🎯 Goals & Motivation

### Short-term Goals (This Week)
- Complete all lectures
- Finish one Jupyter notebook
- Submit assignment on time
- Solve 10 practice problems

### Medium-term Goals (This Semester)
- Master all foundation topics
- Build 3 mini-projects
- Score well in exams
- Help classmates

### Long-term Goals (Program)
- Complete BS degree
- Master Data Science skills
- Build impressive portfolio
- Career in Data Science/ML

---

## 💡 Pro Tips

1. **Start small** - Don't try to do everything at once
2. **Be consistent** - Daily small steps beat weekend marathons
3. **Stay organized** - Use this folder structure religiously
4. **Collaborate** - Study with peers, but do your own work
5. **Take breaks** - Pomodoro technique works wonders
6. **Sleep well** - Your brain needs rest to consolidate learning
7. **Ask questions** - No question is stupid
8. **Celebrate wins** - Acknowledge your progress

## 📞 Support Resources

- **IIT Madras Portal**: https://ds.study.iitm.ac.in/
- **Discussion Forum**: https://discourse.onlinedegree.iitm.ac.in/
- **Email Support**: support@study.iitm.ac.in
- **Phone**: 7850999966

---

**Created**: November 14, 2025  
**Last Updated**: November 14, 2025

**Remember**: You're not just learning for exams, you're building skills for life! 🚀
