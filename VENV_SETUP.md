# Virtual Environment Setup for IIT Madras BS Learning

## Quick Start

### 1. Activate the Virtual Environment

**Option A: Using the activation script** (Recommended)
```bash
source activate.sh
```

**Option B: Manual activation**
```bash
source .venv/bin/activate
```

### 2. Verify Installation
```bash
python --version          # Should show Python 3.9+
pip list                  # See all installed packages
jupyter --version         # Verify Jupyter is installed
```

### 3. Start Jupyter
```bash
jupyter lab               # Start JupyterLab (recommended)
# or
jupyter notebook          # Start classic Jupyter Notebook
```

### 4. Deactivate When Done
```bash
deactivate
```

---

## What's Installed

### Core Data Science Stack
- **NumPy** - Numerical computing
- **Pandas** - Data manipulation and analysis
- **SciPy** - Scientific computing
- **SymPy** - Symbolic mathematics

### Visualization
- **Matplotlib** - Basic plotting
- **Seaborn** - Statistical visualizations
- **Plotly** - Interactive plots
- **matplotlib-venn** - Venn diagrams

### Jupyter Environment
- **Jupyter Lab** - Modern notebook interface
- **Jupyter Notebook** - Classic notebook interface
- **ipywidgets** - Interactive widgets
- **nbformat** - Notebook format tools

### Machine Learning
- **scikit-learn** - Machine learning algorithms

### Statistics
- **statsmodels** - Statistical models and tests

### Web & Data
- **requests** - HTTP library
- **beautifulsoup4** - Web scraping
- **lxml** - XML/HTML parser

### Development Tools
- **black** - Code formatter
- **pylint** - Code linter
- **autopep8** - PEP 8 formatter

---

## Troubleshooting

### Virtual Environment Not Activating
```bash
# Recreate from scratch
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Package Installation Failed
```bash
# Upgrade pip first
pip install --upgrade pip setuptools wheel
# Then retry
pip install -r requirements.txt
```

### Jupyter Kernel Not Found
```bash
# Install ipykernel and add to Jupyter
pip install ipykernel
python -m ipykernel install --user --name=iitm-bs --display-name="IIT Madras BS"
```

### Import Errors in Notebooks
```bash
# Make sure you're using the correct kernel
# In Jupyter: Kernel > Change Kernel > IIT Madras BS (or Python 3.9)
```

---

## Adding New Packages

### Single Package
```bash
pip install package-name
# Then update requirements.txt
pip freeze > requirements.txt
```

### From requirements.txt
```bash
pip install -r requirements.txt
```

---

## VS Code Integration

### Select Python Interpreter
1. Open Command Palette (Cmd+Shift+P)
2. Type "Python: Select Interpreter"
3. Choose `.venv/bin/python`

### Jupyter in VS Code
- Open any `.ipynb` file
- VS Code will automatically detect the .venv kernel
- Select kernel from top-right dropdown if needed

---

## Best Practices

1. **Always activate before working**
   ```bash
   source .venv/bin/activate
   ```

2. **Keep requirements.txt updated**
   ```bash
   pip freeze > requirements.txt
   ```

3. **Use separate environments for different projects**
   - This `.venv` is specific to iitm-bs-learning
   - Don't mix with other projects

4. **Deactivate when switching projects**
   ```bash
   deactivate
   ```

---

## Common Commands

```bash
# List installed packages
pip list

# Show package info
pip show numpy

# Update a package
pip install --upgrade numpy

# Uninstall a package
pip uninstall package-name

# Export current environment
pip freeze > requirements.txt

# Start Jupyter Lab
jupyter lab

# Run Python script
python script.py

# Run Python in interactive mode
python
```

---

## Environment Variables

The `.venv` automatically sets:
- `VIRTUAL_ENV` - Path to virtual environment
- `PATH` - Includes `.venv/bin` first
- `PS1` - Shows `(.venv)` in prompt

---

## Size and Maintenance

### Check Environment Size
```bash
du -sh .venv
# Typically: 200-400 MB
```

### Clean Unused Packages
```bash
pip install pip-autoremove
pip-autoremove package-name -y
```

### Recreate from Scratch
```bash
deactivate                      # Exit if activated
rm -rf .venv                    # Remove old environment
python3 -m venv .venv           # Create new
source .venv/bin/activate       # Activate
pip install -r requirements.txt # Reinstall packages
```

---

## Notes

- `.venv` is in `.gitignore` (not tracked by Git)
- Other users should create their own `.venv`
- Share `requirements.txt`, not the `.venv` folder
- Python version: 3.9+ (compatible with all packages)

---

**Created**: 2025-11-15  
**Python Version**: 3.9.6  
**Total Packages**: 40+
