"""
IIT Madras BS Data Science - Course Data Fetcher
Fetches course information from IIT Madras website and generates structured notes and notebooks

Usage:
    python fetch_course_data.py --course BSMA1001  # Fetch specific course
    python fetch_course_data.py --level foundation  # Fetch all foundation courses
    python fetch_course_data.py --all              # Fetch all courses

Features:
- Fetches course metadata (name, code, credits, instructors)
- Extracts week-by-week syllabus
- Generates structured Markdown notes
- Creates Jupyter notebook templates
- Organizes files in appropriate folders
"""

import requests
from bs4 import BeautifulSoup
import os
import json
from pathlib import Path
from datetime import datetime
import re
import argparse

class IITMCourseFetcher:
    """Fetch and process IIT Madras BS Data Science course information"""
    
    BASE_URL = "https://study.iitm.ac.in/ds/course_pages/"
    
    # Course code to folder mapping
    COURSE_MAPPING = {
        # Foundation Level
        'BSMA1001': ('01-Foundation-Level/01-Mathematics', 'Mathematics for Data Science I'),
        'BSMA1002': ('01-Foundation-Level/02-Statistics', 'Statistics for Data Science I'),
        'BSCS1001': ('01-Foundation-Level/04-Computational-Thinking', 'Computational Thinking'),
        'BSHS1001': ('01-Foundation-Level/05-English', 'English I'),
        'BSMA1003': ('01-Foundation-Level/01-Mathematics-II', 'Mathematics for Data Science II'),
        'BSMA1004': ('01-Foundation-Level/02-Statistics-II', 'Statistics for Data Science II'),
        'BSCS1002': ('01-Foundation-Level/03-Python-Programming', 'Programming in Python'),
        'BSHS1002': ('01-Foundation-Level/05-English', 'English II'),
    }
    
    def __init__(self, base_path=None):
        """Initialize fetcher with base repository path"""
        if base_path is None:
            # Assume script is in scripts/ folder
            self.base_path = Path(__file__).parent.parent
        else:
            self.base_path = Path(base_path)
            
    def fetch_course_page(self, course_code):
        """Fetch course page HTML"""
        url = f"{self.BASE_URL}{course_code}.html"
        print(f"Fetching: {url}")
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            print(f"Error fetching {course_code}: {e}")
            return None
    
    def parse_course_data(self, html, course_code):
        """Parse course information from HTML"""
        soup = BeautifulSoup(html, 'html.parser')
        
        course_data = {
            'code': course_code,
            'name': '',
            'description': '',
            'credits': '',
            'level': '',
            'prerequisites': '',
            'instructors': [],
            'learning_objectives': [],
            'weeks': [],
            'references': [],
            'textbooks': [],
            'youtube_link': '',
            'fetched_date': datetime.now().isoformat()
        }
        
        # Extract course name
        name_elem = soup.find('h1') or soup.find('h2')
        if name_elem:
            course_data['name'] = name_elem.get_text(strip=True)
        
        # Extract description
        desc_elem = soup.find('p', class_='course-description') or soup.find_all('p')
        if desc_elem:
            if isinstance(desc_elem, list):
                course_data['description'] = desc_elem[0].get_text(strip=True)
            else:
                course_data['description'] = desc_elem.get_text(strip=True)
        
        # Extract metadata
        info_items = soup.find_all(['p', 'div'], class_=re.compile('course-info|meta'))
        for item in info_items:
            text = item.get_text()
            if 'Course ID:' in text:
                course_data['code'] = text.split('Course ID:')[1].strip()
            if 'Credits:' in text:
                course_data['credits'] = text.split('Credits:')[1].strip()
            if 'Level' in text or 'Type:' in text:
                course_data['level'] = text.split(':')[1].strip() if ':' in text else ''
            if 'Pre-requisites:' in text or 'Prerequisites:' in text:
                course_data['prerequisites'] = text.split(':')[1].strip()
        
        # Extract learning objectives
        objectives_section = soup.find(text=re.compile('What you\'ll learn|Learning Objectives'))
        if objectives_section:
            parent = objectives_section.find_parent()
            if parent:
                objectives = parent.find_all(['li', 'p'])
                course_data['learning_objectives'] = [obj.get_text(strip=True) for obj in objectives if obj.get_text(strip=True)]
        
        # Extract week-by-week syllabus from table
        week_table = soup.find('table')
        if week_table:
            rows = week_table.find_all('tr')
            for row in rows:
                cols = row.find_all('td')
                if len(cols) >= 2:
                    week_num = cols[0].get_text(strip=True)
                    topics = cols[1].get_text(strip=True)
                    course_data['weeks'].append({
                        'week': week_num,
                        'topics': topics
                    })
        
        # Extract YouTube playlist link
        youtube_link = soup.find('a', href=re.compile('youtube.com|youtu.be'))
        if youtube_link:
            course_data['youtube_link'] = youtube_link.get('href', '')
        
        # Extract instructors
        instructor_divs = soup.find_all(['div', 'p'], class_=re.compile('instructor'))
        for div in instructor_divs:
            name = div.find(['h3', 'h4', 'strong'])
            if name:
                course_data['instructors'].append(name.get_text(strip=True))
        
        return course_data
    
    def generate_notes_markdown(self, course_data):
        """Generate structured Markdown notes from course data"""
        course_code = course_data['code']
        course_name = course_data['name']
        
        markdown = f"""# {course_name}

**Course Code**: {course_code}  
**Credits**: {course_data['credits']}  
**Level**: {course_data['level']}  
**Prerequisites**: {course_data['prerequisites']}  
**Date Created**: {datetime.now().strftime('%Y-%m-%d')}

## Course Description

{course_data['description']}

## Learning Objectives

"""
        
        for i, obj in enumerate(course_data['learning_objectives'], 1):
            markdown += f"{i}. {obj}\n"
        
        markdown += f"\n## Instructors\n\n"
        for instructor in course_data['instructors']:
            markdown += f"- {instructor}\n"
        
        markdown += f"\n## Resources\n\n"
        if course_data['youtube_link']:
            markdown += f"- [Course Videos]({course_data['youtube_link']})\n"
        
        markdown += f"\n## Week-by-Week Syllabus\n\n"
        
        for week in course_data['weeks']:
            week_num = week['week']
            topics = week['topics']
            markdown += f"### {week_num}\n\n"
            markdown += f"**Topics**: {topics}\n\n"
            markdown += f"#### Key Concepts\n"
            markdown += f"<!-- Add key concepts here -->\n\n"
            markdown += f"#### Definitions\n"
            markdown += f"<!-- Add definitions here -->\n\n"
            markdown += f"#### Important Formulas\n"
            markdown += f"<!-- Add formulas here -->\n\n"
            markdown += f"#### Examples\n"
            markdown += f"<!-- Add worked examples here -->\n\n"
            markdown += f"#### Questions/Doubts\n"
            markdown += f"- [ ] <!-- Add questions here -->\n\n"
            markdown += f"---\n\n"
        
        markdown += f"\n## Study Notes\n\n"
        markdown += f"<!-- Add your personal study notes below -->\n\n"
        
        markdown += f"\n## References\n\n"
        for ref in course_data['references']:
            markdown += f"- {ref}\n"
        
        markdown += f"\n---\n\n"
        markdown += f"**Last Updated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n"
        markdown += f"**Data Source**: IIT Madras BS Data Science Program  \n"
        
        return markdown
    
    def generate_notebook_template(self, course_data, week_num=1):
        """Generate Jupyter notebook template for a specific week"""
        import nbformat
        from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
        
        course_code = course_data['code']
        course_name = course_data['name']
        
        # Get week data
        week_data = None
        for week in course_data['weeks']:
            if f"WEEK {week_num}" in week['week'].upper() or f"Week {week_num}" in week['week']:
                week_data = week
                break
        
        if not week_data:
            week_data = {'week': f'Week {week_num}', 'topics': 'Topics to be added'}
        
        nb = new_notebook()
        
        # Header cell
        header = f"""# {course_name} - {week_data['week']}

**Course Code**: {course_code}  
**Date**: {datetime.now().strftime('%Y-%m-%d')}  
**Topics**: {week_data['topics']}

**Prerequisites**:
- Review notes/{week_data['week'].lower().replace(' ', '-')}.md first
- Ensure Python environment is set up

**Learning Goals**:
- Implement concepts from lecture notes
- Practice with code examples
- Visualize key concepts
- Solve computational problems
"""
        nb.cells.append(new_markdown_cell(header))
        
        # Setup cell
        setup_code = """# Setup: Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("✓ Libraries imported successfully")
"""
        nb.cells.append(new_code_cell(setup_code))
        
        # Example sections
        sections = [
            ("Example 1: Basic Concepts", "# TODO: Implement basic concept from notes\n\n"),
            ("Visualization", "# TODO: Create visualization\n\n"),
            ("Example 2: Real-World Application", "# TODO: Apply concept to real problem\n\n"),
            ("Practice Problems", "# TODO: Solve practice problems\n# Problem 1:\n\n# Problem 2:\n\n"),
            ("Experiments", "# TODO: Experiment with variations\n\n"),
        ]
        
        for section_title, code_template in sections:
            nb.cells.append(new_markdown_cell(f"## {section_title}\n"))
            nb.cells.append(new_code_cell(code_template))
        
        # Summary cell
        summary = """## Key Takeaways

**What I learned**:
- 
- 
- 

**Insights from coding**:
- 
- 

**Questions for next session**:
- 
- 

**Next steps**:
- [ ] Review this week's notes
- [ ] Complete practice problems
- [ ] Start next week's material
"""
        nb.cells.append(new_markdown_cell(summary))
        
        return nb
    
    def save_course_materials(self, course_data):
        """Save generated notes and notebooks to appropriate folders"""
        course_code = course_data['code']
        
        # Get folder mapping
        if course_code not in self.COURSE_MAPPING:
            print(f"Warning: No folder mapping for {course_code}")
            return
        
        folder_path, _ = self.COURSE_MAPPING[course_code]
        course_folder = self.base_path / folder_path
        
        # Create folders if they don't exist
        notes_folder = course_folder / 'notes'
        notebooks_folder = course_folder / 'notebooks'
        notes_folder.mkdir(parents=True, exist_ok=True)
        notebooks_folder.mkdir(parents=True, exist_ok=True)
        
        # Generate and save overview notes
        overview_markdown = self.generate_notes_markdown(course_data)
        overview_file = notes_folder / f'00-{course_code}-overview.md'
        overview_file.write_text(overview_markdown, encoding='utf-8')
        print(f"✓ Created: {overview_file}")
        
        # Generate week-specific notes templates
        for i, week_data in enumerate(course_data['weeks'], 1):
            week_file = notes_folder / f'week-{i:02d}-notes.md'
            if not week_file.exists():  # Don't overwrite existing notes
                week_markdown = f"""# Week {i}: {week_data['topics']}

**Date**: {datetime.now().strftime('%Y-%m-%d')}  
**Course**: {course_data['name']} ({course_code})

## Topics Covered

{week_data['topics']}

## Key Concepts

### Concept 1
<!-- Add concept explanation -->

### Concept 2
<!-- Add concept explanation -->

## Definitions

- **Term 1**: Definition
- **Term 2**: Definition

## Important Formulas

- Formula 1: 
- Formula 2: 

## Theorems & Proofs

### Theorem 1
<!-- Add theorem and proof -->

## Examples (Worked Problems)

### Example 1
**Problem**: 
**Solution**: 

## Questions/Doubts

- [ ] Question 1
- [ ] Question 2

## Action Items

- [ ] Review lecture slides
- [ ] Complete practice problems
- [ ] Work through notebook examples

## References

- Textbook: Chapter X
- Lecture video: [Link]

---

**Next Class**: Week {i+1}
"""
                week_file.write_text(week_markdown, encoding='utf-8')
                print(f"✓ Created: {week_file}")
        
        # Generate notebooks for first 3 weeks (examples)
        import nbformat
        for i in range(1, min(4, len(course_data['weeks']) + 1)):
            notebook_file = notebooks_folder / f'week-{i:02d}-practice.ipynb'
            if not notebook_file.exists():  # Don't overwrite existing notebooks
                nb = self.generate_notebook_template(course_data, i)
                with open(notebook_file, 'w', encoding='utf-8') as f:
                    nbformat.write(nb, f)
                print(f"✓ Created: {notebook_file}")
        
        # Save raw course data as JSON for reference
        data_folder = course_folder / 'resources'
        data_folder.mkdir(parents=True, exist_ok=True)
        json_file = data_folder / f'{course_code}_data.json'
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(course_data, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved course data: {json_file}")
    
    def process_course(self, course_code):
        """Full pipeline: fetch, parse, generate, and save"""
        print(f"\n{'='*60}")
        print(f"Processing {course_code}")
        print(f"{'='*60}")
        
        # Fetch HTML
        html = self.fetch_course_page(course_code)
        if not html:
            return False
        
        # Parse data
        course_data = self.parse_course_data(html, course_code)
        print(f"✓ Parsed: {course_data['name']}")
        print(f"  - {len(course_data['weeks'])} weeks")
        print(f"  - {len(course_data['learning_objectives'])} learning objectives")
        
        # Generate and save materials
        self.save_course_materials(course_data)
        
        return True

def main():
    parser = argparse.ArgumentParser(description='Fetch IIT Madras course data and generate study materials')
    parser.add_argument('--course', help='Course code (e.g., BSMA1001)')
    parser.add_argument('--level', choices=['foundation', 'diploma', 'bsc', 'bs', 'all'], help='Fetch all courses at a level')
    parser.add_argument('--base-path', help='Base repository path (default: parent of script folder)')
    
    args = parser.parse_args()
    
    fetcher = IITMCourseFetcher(args.base_path)
    
    # Define course groups
    foundation_courses = ['BSMA1001', 'BSMA1002', 'BSCS1001', 'BSHS1001', 
                         'BSMA1003', 'BSMA1004', 'BSCS1002', 'BSHS1002']
    
    if args.course:
        # Fetch single course
        fetcher.process_course(args.course)
    elif args.level == 'foundation':
        # Fetch all foundation courses
        for course_code in foundation_courses:
            fetcher.process_course(course_code)
            print()
    elif args.level == 'all':
        print("Fetching all courses - this will take a while...")
        # Read all URLs from file
        urls_file = Path(__file__).parent / 'course_urls.txt'
        with open(urls_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and 'course_pages' in line:
                    course_code = line.split('/')[-1].replace('.html', '')
                    fetcher.process_course(course_code)
                    print()
    else:
        print("Please specify --course or --level")
        print("Example: python fetch_course_data.py --course BSMA1001")
        print("Example: python fetch_course_data.py --level foundation")

if __name__ == '__main__':
    main()
