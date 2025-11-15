# RAG Learning & Notes

Welcome to my Retrieval-Augmented Generation (RAG) learning repository! This workspace contains my studies, experiments, and notes on RAG systems as part of my journey in the **IIT Madras BS in Data Science and Applications** program.

## 🎓 About My Studies

I am currently enrolled in the [IIT Madras BS in Data Science and Applications](https://study.iitm.ac.in/ds/) program - a comprehensive 4-year online degree program covering:
- **Data Science Courses**: ML Foundations, Deep Learning, Reinforcement Learning, Computer Vision, LLMs, Big Data
- **Programming**: Python, Java, PostgreSQL, Linux, C Programming, Full Stack Development
- **Frameworks**: Flask, Vue, NumPy, Scikit-learn, PyTorch, OpenCV, Kafka
- **Business Analytics**: Business Data Management, Market Research, Managerial Economics

This repository focuses specifically on my RAG (Retrieval-Augmented Generation) learning and experimentation.

## 📚 What is RAG?

Retrieval-Augmented Generation (RAG) is a technique that enhances Large Language Models (LLMs) by combining them with external knowledge retrieval systems. Instead of relying solely on the model's training data, RAG:

1. **Retrieves** relevant information from a knowledge base
2. **Augments** the prompt with this context
3. **Generates** responses based on both the retrieved information and the model's capabilities

## 🎯 Learning Objectives

- [ ] Understand core RAG concepts and architecture
- [ ] Learn about vector databases and embeddings
- [ ] Implement basic RAG pipelines
- [ ] Explore advanced RAG techniques (HyDE, Multi-query, etc.)
- [ ] Study evaluation metrics for RAG systems
- [ ] Build production-ready RAG applications
- [ ] Apply RAG concepts to IIT Madras coursework (ML, Deep Learning, LLMs)
- [ ] Integrate RAG with frameworks learned in the program (PyTorch, Scikit-learn)

## 📂 Folder Structure

```
Learning/
├── README.md                          # This file (RAG + IIT Madras overview)
├── STUDY-GUIDE.md                     # 📖 Complete study organization guide
├── GETTING-STARTED.md                 # 🚀 Quick start for Foundation level
├── PROGRAM-OVERVIEW.md                # 🎓 Complete program structure (Foundation→MTech)
│
├── 00-RAG-Studies/                    # RAG learning materials
│   ├── notes/                         # RAG concepts and theory
│   ├── experiments/                   # RAG code experiments
│   └── projects/                      # RAG applications
│
├── 01-Foundation-Level/               # � CURRENT LEVEL (32 credits)
│   ├── README.md                      # Foundation overview + study tips
│   ├── 01-Mathematics/                # Math I & II
│   │   ├── notes/ notebooks/ assignments/ practice/ resources/
│   │   └── ✅ Starter notebook included
│   ├── 02-Statistics/                 # Stats I & II  
│   │   ├── notes/ notebooks/ assignments/ practice/ resources/
│   │   └── ✅ Starter notebook included
│   ├── 03-Python-Programming/         # CT + Python
│   │   ├── notes/ notebooks/ assignments/ practice/ projects/ resources/
│   │   └── ✅ Starter notebook included
│   ├── 04-Computational-Thinking/
│   │   └── notes/ notebooks/ assignments/ resources/
│   └── 05-English/                    # English I & II
│       └── notes/ assignments/ resources/
│
├── 02-Diploma-Level/                  # 📚 Diploma (54 credits)
│   ├── README.md                      # Diploma level guide
│   ├── 01-Programming/                # Diploma in Programming (27 cr)
│   │   ├── notes/ notebooks/ assignments/ projects/ resources/
│   │   └── Courses: DBMS, DSA, MAD I/II, Java, Sys Commands
│   └── 02-Data-Science/               # Diploma in Data Science (27 cr)
│       ├── notes/ notebooks/ assignments/ projects/ resources/
│       └── Courses: ML Foundation/Tech/Practice, BDM, Tools, Track choice
│
├── 03-BSc-Degree-Level/               # 🎓 BSc (114 total credits)
│   ├── README.md                      # BSc degree guide
│   ├── notes/ notebooks/ assignments/ projects/ resources/
│   └── Core: Software Eng, Testing, AI Search, Deep Learning + Electives
│
├── 04-BS-Degree-Level/                # 🎓 BS (142 total credits)
│   ├── README.md                      # BS degree guide  
│   ├── notes/ notebooks/ assignments/ projects/ resources/
│   └── Advanced Electives: LLMs, NLP, CV, RL, MLOps, GenAI, etc.
│
├── 05-PG-Diploma-Level/               # 🎓 PG Diploma in AI & ML (162 total)
│   ├── README.md                      # PG Diploma guide
│   ├── notes/ notebooks/ assignments/ projects/ resources/
│   └── Core: MLOps, GenAI Math, Algorithms + 2 Electives
│
├── 06-MTech-Level/                    # 🎓 MTech in AI & ML (182 total)
│   ├── README.md                      # MTech guide
│   ├── project/                       # MTech research project (20 cr)
│   ├── research/                      # Research papers and notes
│   ├── notes/
│   └── resources/
│
└── 99-Resources/                      # Shared resources
    ├── templates/                     # Note templates
    ├── cheatsheets/                   # Quick references
    └── datasets/                      # Practice datasets
```

## 🎯 Program Path (Foundation → MTech)

**Your Journey**: 182 credits over 4-8 years

1. **Foundation** (32cr) → Foundation Certificate
2. **Diploma** (54cr) → Diploma(s) in Programming &/or Data Science
3. **BSc** (28cr) → BSc in Programming & Data Science  
4. **BS** (28cr) → BS in Data Science & Applications
5. **PG Diploma** (20cr) → PG Diploma in AI & ML (requires CGPA ≥ 8.0)
6. **MTech** (20cr) → BS + MTech in AI & ML

**📘 See [PROGRAM-OVERVIEW.md](./PROGRAM-OVERVIEW.md) for complete details!**  
**📘 See [STUDY-GUIDE.md](./STUDY-GUIDE.md) for organization strategies!**  
**📘 See [GETTING-STARTED.md](./GETTING-STARTED.md) to begin your Foundation level!**

## 🛠️ Key Technologies

- **LLMs**: OpenAI GPT, Anthropic Claude, Open-source models (Llama, Mistral)
- **Vector Databases**: Pinecone, Weaviate, Chroma, FAISS, Qdrant
- **Embeddings**: OpenAI embeddings, Sentence Transformers, Cohere
- **Frameworks**: LangChain, LlamaIndex, Haystack
- **Languages**: Python, TypeScript/JavaScript

## 📖 Topics to Cover

### Fundamentals
- Vector embeddings and similarity search
- Chunking strategies
- Prompt engineering for RAG
- Context window management

### Advanced Topics
- Hybrid search (dense + sparse)
- Re-ranking strategies
- Query transformation techniques
- Multi-modal RAG
- Graph RAG
- Agentic RAG

### Evaluation & Optimization
- RAGAS framework
- Faithfulness and relevance metrics
- Latency optimization
- Cost optimization

## 🔗 Resources

### Essential Readings
- [RAG Survey Paper](https://arxiv.org/abs/2312.10997)
- [LangChain RAG Documentation](https://python.langchain.com/docs/use_cases/question_answering/)
- [LlamaIndex RAG Guide](https://docs.llamaindex.ai/)

### Courses & Tutorials
- Add your course links here
- Add tutorial resources here

### Communities
- LangChain Discord
- LlamaIndex Discord
- r/LocalLLaMA

## 📝 Notes Format

For consistency, structure your notes using:

```markdown
# Topic Name

## Date: YYYY-MM-DD

## Summary
Brief overview of what was learned

## Key Concepts
- Concept 1: Description
- Concept 2: Description

## Implementation Details
Code snippets, examples, or technical details

## References
Links to sources, papers, or documentation

## Next Steps
What to explore next
```

## 🚀 Getting Started

### Prerequisites
```bash
# Python environment
python -m venv venv
source venv/bin/activate  # On macOS/Linux

# Install common dependencies
pip install langchain openai chromadb sentence-transformers
```

### Quick Start
1. Clone or navigate to this directory
2. Set up your Python environment
3. Create a `.env` file for API keys
4. Start with `experiments/basic-rag/` for simple examples

## 🔐 Environment Variables

Create a `.env` file in the root directory:

```env
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
COHERE_API_KEY=your_key_here
PINECONE_API_KEY=your_key_here
```

## 📊 Progress Tracker

| Topic | Status | Notes |
|-------|--------|-------|
| RAG Fundamentals | 🔄 In Progress | |
| Vector Databases | 📝 Planned | |
| Advanced Techniques | 📝 Planned | |
| Production Deployment | 📝 Planned | |

**Legend**: ✅ Complete | 🔄 In Progress | 📝 Planned

## 💡 Project Ideas

- [ ] Document QA chatbot
- [ ] Personal knowledge base assistant
- [ ] Code documentation search
- [ ] Research paper analyzer
- [ ] Multi-lingual RAG system
- [ ] **IIT Madras Course Assistant**: RAG system for course materials and lecture notes
- [ ] **Study Buddy Bot**: Q&A system for exam preparation
- [ ] **Assignment Helper**: Context-aware coding assistant for coursework

## 📅 Learning Log

Keep a log of your learning journey:

- **2025-11-14**: Started RAG learning repository

---

## 🤝 Contributing to This Repo

This is a personal learning repository, but feel free to:
- Add new notes and experiments
- Improve documentation
- Share interesting findings
- Document challenges and solutions

## 📧 Contact

**IIT Madras BS Data Science Student**  
Program: [BS in Data Science and Applications](https://study.iitm.ac.in/ds/)

---

**Last Updated**: November 14, 2025
