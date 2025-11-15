# Course Data Fetcher Scripts

Automated tools to fetch course information from IIT Madras BS Data Science program and generate structured study materials.

## 📁 Files

- **`course_urls.txt`**: Complete list of all IIT Madras BS course URLs
- **`fetch_course_data.py`**: Main script to fetch and generate materials
- **`requirements.txt`**: Python dependencies

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd /Users/godsonkurishinkal/Projects/iitm-bs-learning/scripts
pip install -r requirements.txt
```

### 2. Fetch Course Data

**Fetch a single course**:
```bash
python fetch_course_data.py --course BSMA1001
```

**Fetch all Foundation level courses**:
```bash
python fetch_course_data.py --level foundation
```

**Fetch all courses** (will take time):
```bash
python fetch_course_data.py --level all
```

## 📚 What Gets Generated

For each course, the script creates:

### 1. Overview Notes (`notes/00-[COURSE]-overview.md`)
- Course description and objectives
- Instructor information
- Week-by-week syllabus overview
- Resources and references

### 2. Weekly Notes Templates (`notes/week-XX-notes.md`)
- Pre-structured Markdown templates
- Sections for concepts, definitions, formulas
- Space for examples and questions
- Following the repository's note-taking format

### 3. Jupyter Notebooks (`notebooks/week-XX-practice.ipynb`)
- Code templates for each week
- Setup cells with common imports
- Example sections to implement concepts
- Practice problem placeholders
- Summary sections for reflection

### 4. Course Data JSON (`resources/[COURSE]_data.json`)
- Raw structured data for reference
- Can be used for further processing

## 🎯 Example Output Structure

After running for BSMA1001:

```
01-Foundation-Level/01-Mathematics/
├── notes/
│   ├── 00-BSMA1001-overview.md          # ✅ Generated
│   ├── week-01-notes.md                  # ✅ Generated template
│   ├── week-02-notes.md                  # ✅ Generated template
│   └── ...
├── notebooks/
│   ├── week-01-practice.ipynb            # ✅ Generated template
│   ├── week-02-practice.ipynb            # ✅ Generated template
│   └── ...
└── resources/
    └── BSMA1001_data.json                # ✅ Generated data
```

## 🔧 How It Works

1. **Fetches** course page HTML from IIT Madras website
2. **Parses** course information using BeautifulSoup:
   - Course metadata (code, credits, level)
   - Learning objectives
   - Week-by-week syllabus
   - Instructor information
   - Resource links
3. **Generates** structured materials:
   - Markdown notes following repository template
   - Jupyter notebooks with proper structure
   - JSON data for reference
4. **Organizes** files into correct folders based on course level

## 📝 Course Code Mapping

The script automatically maps course codes to folders:

| Course Code | Folder | Course Name |
|-------------|--------|-------------|
| BSMA1001 | 01-Foundation-Level/01-Mathematics | Mathematics for DS I |
| BSMA1002 | 01-Foundation-Level/02-Statistics | Statistics for DS I |
| BSCS1001 | 01-Foundation-Level/04-Computational-Thinking | Computational Thinking |
| BSHS1001 | 01-Foundation-Level/05-English | English I |
| BSMA1003 | 01-Foundation-Level/01-Mathematics-II | Mathematics for DS II |
| BSMA1004 | 01-Foundation-Level/02-Statistics-II | Statistics for DS II |
| BSCS1002 | 01-Foundation-Level/03-Python-Programming | Programming in Python |
| BSHS1002 | 01-Foundation-Level/05-English | English II |

## ⚙️ Configuration

### Custom Base Path
```bash
python fetch_course_data.py --course BSMA1001 --base-path /path/to/repo
```

### Course URL Format
Course pages follow pattern: `https://study.iitm.ac.in/ds/course_pages/[CODE].html`

## 🎓 Generated Content Format

### Notes Format
- Follows repository's Markdown template
- Includes all required sections (concepts, definitions, formulas)
- Ready for AI assistant context understanding
- Week-by-week structured for progressive learning

### Notebook Format
- Header with course info and prerequisites
- Setup cell with standard imports
- Multiple example sections
- Practice problem placeholders
- Summary and reflection sections

## 🚨 Important Notes

1. **Don't Overwrite**: Script won't overwrite existing notes/notebooks
2. **Manual Content**: Generated files are templates - add your actual study content
3. **Week Data**: Some courses may have limited week information on website
4. **Network Required**: Needs internet connection to fetch course pages
5. **Respect Website**: Script includes reasonable delays between requests

## 🛠️ Troubleshooting

**Import Error**: Install dependencies with `pip install -r requirements.txt`

**No Course Data**: Check if course URL is correct in `course_urls.txt`

**Permission Error**: Ensure script has write access to repository folders

**Network Error**: Check internet connection and website accessibility

## 📈 Future Enhancements

- [ ] Add support for scraping full 12-week syllabus (requires JavaScript)
- [ ] Generate practice problems from course content
- [ ] Create flashcards for definitions and formulas
- [ ] Add progress tracking integration
- [ ] Support for updating existing materials

## 🤝 Contributing

This script is part of the personal learning repository. Modify as needed for your use case.

---

**Created**: 2025-11-15  
**Purpose**: Automate study material generation for IIT Madras BS Data Science program
