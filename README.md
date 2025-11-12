# LangChain Learning Journey 📚

A comprehensive repository documenting my learning progress with LangChain, covering LLMs, Chat Models, Embeddings, and more.

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [Learning Progress](#learning-progress)
  - [1. LLMs (Language Models)](#1-llms-language-models)
  - [2. Chat Models](#2-chat-models)
  - [3. Embedding Models](#3-embedding-models)
- [Requirements](#requirements)
- [Environment Setup](#environment-setup)
- [Usage Examples](#usage-examples)
- [Notes & Learnings](#notes--learnings)
- [Future Plans](#future-plans)

---

## 🎯 Overview

This repository contains practical examples and implementations as I learn LangChain. Each module demonstrates different aspects of working with Large Language Models (LLMs) and their applications.

**Current Status:** 🟢 Active Learning

---

## 📁 Project Structure

```
Langchain-Models/
├── 1.LLMs/                    # Basic Language Model implementations
│   └── demo_openai.py         # OpenAI LLM demo
├── 2.ChatModels/             # Chat Model implementations
│   ├── chat-demo-openai.py   # OpenAI Chat Model
│   ├── chat-demo-anthropic.py # Anthropic Chat Model
│   ├── chat-gemini.py         # Google Gemini Chat Model
│   ├── chatmodel-huggingface.py # Hugging Face Chat Model (API)
│   └── chat-hf-localhosted.py # Hugging Face Chat Model (Local)
├── 3.EmbeddingModels/        # Embedding Model implementations (Coming Soon)
├── requirements.md            # Package dependencies
├── .env                       # Environment variables (API keys)
└── README.md                  # This file
```

---

## 🚀 Setup Instructions

### Prerequisites

- Python 3.12+ (recommended) or Python 3.13+
- pip package manager
- Git (for version control)

### Installation Steps

1. **Clone the repository** (if applicable)

   ```bash
   git clone <repository-url>
   cd Langchain-Models
   ```

2. **Create a virtual environment**

   ```bash
   # Using Python 3.12 (recommended for LangChain compatibility)
   py -3.12 -m venv venv

   # Or using default Python
   python -m venv venv
   ```

3. **Activate the virtual environment**

   ```powershell
   # Windows PowerShell
   .\venv\Scripts\Activate.ps1

   # Windows CMD
   venv\Scripts\activate.bat

   # macOS/Linux
   source venv/bin/activate
   ```

4. **Install dependencies**

   ```bash
   pip install -r requirements.md
   ```

   Or install individually:

   ```bash
   pip install langchain langchain-core
   pip install langchain-openai
   pip install langchain-anthropic
   pip install langchain-google-genai
   pip install langchain-huggingface
   pip install python-dotenv
   ```

5. **Set up environment variables**

   Create a `.env` file in the root directory:

   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   ANTHROPIC_API_KEY=your_anthropic_api_key_here
   GOOGLE_API_KEY=your_google_api_key_here
   HUGGINGFACEHUB_API_TOKEN=your_huggingface_token_here
   ```

---

## 📚 Learning Progress

### 1. LLMs (Language Models)

**Status:** ✅ Completed

Basic Language Model implementations for simple text generation tasks.

#### Implementations:

- **OpenAI LLM** (`1.LLMs/demo_openai.py`)
  - Model: `gpt-3.5-turbo-instruct`
  - Use case: Simple text generation
  - Example: Question answering

**Key Learnings:**

- Difference between LLMs and Chat Models
- Basic invocation patterns
- Environment variable management with `python-dotenv`

---

### 2. Chat Models

**Status:** ✅ In Progress

Chat Models are designed for conversational interactions with structured message formats.

#### Implementations:

1. **OpenAI Chat Model** (`2.ChatModels/chat-demo-openai.py`)

   - Model: `gpt-4`
   - Features: Temperature control, max tokens
   - Status: ✅ Working

2. **Google Gemini Chat Model** (`2.ChatModels/chat-gemini.py`)

   - Model: `gemini-2.5-flash`
   - Features: Temperature control, max tokens
   - Status: ✅ Working

3. **Anthropic Chat Model** (`2.ChatModels/chat-demo-anthropic.py`)

   - Provider: Anthropic Claude
   - Status: 🔄 To be implemented

4. **Hugging Face Chat Model (API)** (`2.ChatModels/chatmodel-huggingface.py`)
   - Model: `openai/gpt-oss-120b` (or other chat-completion compatible models)
   - Features: Chat completion via Hugging Face Inference API
   - Status: ✅ Working
   - Note: Requires `HUGGINGFACEHUB_API_TOKEN` in `.env`
   - Important: Not all models support chat completion - use models like `microsoft/Phi-3-mini-4k-instruct`

5. **Hugging Face Chat Model (Local)** (`2.ChatModels/chat-hf-localhosted.py`)
   - Model: `TinyLlama/TinyLlama-1.1B-Chat-v0.6`
   - Features: Runs models locally using PyTorch
   - Status: ✅ Working
   - Requirements: PyTorch, transformers
   - Note: First run downloads model (~2-4GB). Uses CPU by default.

**Key Learnings:**

- Chat Models vs LLMs: Structured message handling
- Different providers and their APIs
- Model parameter tuning (temperature, max_tokens)
- Provider-specific requirements and compatibility

---

### 3. Embedding Models

**Status:** 🔄 Planned

Embedding models for converting text into vector representations.

**Planned Topics:**

- Text embeddings
- Vector similarity search
- Semantic search applications

---

## 📦 Requirements

See `requirements.md` for the complete list of dependencies.

**Core Packages:**

- `langchain` - Main LangChain library
- `langchain-core` - Core LangChain functionality
- `python-dotenv` - Environment variable management

**Provider Integrations:**

- `langchain-openai` - OpenAI integration
- `langchain-anthropic` - Anthropic integration
- `langchain-google-genai` - Google Gemini integration
- `langchain-huggingface` - Hugging Face integration

**For Local Model Hosting:**

- `torch` - PyTorch (required for local Hugging Face models)
- `transformers` - Hugging Face Transformers library

---

## 🔧 Environment Setup

### Python Version Compatibility

**Important:** LangChain packages currently require Python 3.12 or earlier for full compatibility.

- ✅ Python 3.12 (Recommended)
- ⚠️ Python 3.14+ (May have compatibility issues with Pydantic V1)

**Managing Multiple Python Versions:**

```powershell
# List available Python versions
py --list

# Create venv with specific version
py -3.12 -m venv venv
```

### Environment Variables

All API keys should be stored in `.env` file (which is gitignored):

```env
# OpenAI
OPENAI_API_KEY=sk-...

# Anthropic
ANTHROPIC_API_KEY=sk-ant-...

# Google
GOOGLE_API_KEY=AIzaSy...

# Hugging Face
HUGGINGFACEHUB_API_TOKEN=hf_...
```

**Note:** Never commit `.env` files to version control!

---

## 💡 Usage Examples

### Running LLM Examples

```bash
# Activate virtual environment first
.\venv\Scripts\Activate.ps1

# Run OpenAI LLM demo
python .\1.LLMs\demo_openai.py
```

### Running Chat Model Examples

```bash
# OpenAI Chat Model
python .\2.ChatModels\chat-demo-openai.py

# Google Gemini Chat Model
python .\2.ChatModels\chat-gemini.py

# Hugging Face Chat Model (API)
python .\2.ChatModels\chatmodel-huggingface.py

# Hugging Face Chat Model (Local - requires PyTorch)
python .\2.ChatModels\chat-hf-localhosted.py
```

---

## 📝 Notes & Learnings

### Important Discoveries

1. **Python Version Compatibility**

   - LangChain 1.0.4 requires Python ≤ 3.13
   - Python 3.14+ has issues with Pydantic V1 dependencies
   - Solution: Use Python 3.12 for LangChain projects

2. **Environment Variable Naming**

   - Hugging Face uses `HUGGINGFACEHUB_API_TOKEN` (not `ACCESS_TOKEN`)
   - Always verify environment variable names match library expectations
   - Environment variables cannot contain hyphens: use `HF_HOME` not `HF-HOME`

3. **Model Compatibility**

   - Not all models support chat completion via Hugging Face Inference API
   - Use models like `microsoft/Phi-3-mini-4k-instruct` for chat completion
   - Check model documentation for supported tasks

4. **Virtual Environment Best Practices**
   - Always activate venv before running scripts
   - Use `python-dotenv` to load environment variables
   - Keep `.env` files out of version control

5. **Local Model Hosting**
   - Requires PyTorch for running models locally
   - First run downloads model files (can be several GB)
   - CPU-only PyTorch works but is slower than GPU
   - Set `HF_HOME` environment variable to specify model cache directory

---

## 🐛 Issues Encountered & Solutions

This section documents real issues we faced during development and how we resolved them.

### Issue 1: Python 3.14 Compatibility Problem

**Problem:**
```
ModuleNotFoundError: No module named 'langchain_core.pydantic_v1'
```

**Root Cause:**
- Python 3.14 is too new for LangChain 1.0.4
- Pydantic V1 (used by LangChain) doesn't support Python 3.14+
- Error showed corrupted module names (e.g., `1angchain` instead of `langchain`)

**Solution:**
- Installed Python 3.12 alongside Python 3.14
- Created virtual environment with Python 3.12: `py -3.12 -m venv venv`
- Used `py` launcher to manage multiple Python versions

**Lesson Learned:**
- Always check Python version compatibility before starting a project
- Use `py --list` to see available Python versions on Windows

---

### Issue 2: Missing Python Packages

**Problem:**
```
ModuleNotFoundError: No module named 'dotenv'
ModuleNotFoundError: No module named 'langchain_google_genai'
```

**Root Cause:**
- Packages not installed in virtual environment
- Virtual environment not activated
- IDE using wrong Python interpreter

**Solution:**
- Installed missing packages: `pip install python-dotenv langchain-google-genai`
- Verified virtual environment activation
- Configured IDE to use venv's Python interpreter

**Lesson Learned:**
- Always activate virtual environment before installing packages
- Verify IDE is using correct Python interpreter
- Use `pip list` to check installed packages

---

### Issue 3: Hugging Face API Token Configuration

**Problem:**
```
ValueError: You must provide an api_key to work with novita API
```

**Root Cause:**
- `.env` file had `HUGGINGFACEHUB_ACCESS_TOKEN` instead of `HUGGINGFACEHUB_API_TOKEN`
- Environment variable name mismatch

**Solution:**
- Updated `.env` file to use correct variable name: `HUGGINGFACEHUB_API_TOKEN`
- Verified variable is loaded: `python -c "import os; from dotenv import load_dotenv; load_dotenv(); print(os.getenv('HUGGINGFACEHUB_API_TOKEN'))"`

**Lesson Learned:**
- Always check exact environment variable names in library documentation
- Verify environment variables are loaded correctly

---

### Issue 4: Hugging Face Model Compatibility

**Problem:**
```
StopIteration: No suitable provider found for chat completion
```

**Root Cause:**
- Model `TinyLlama/TinyLlama-1.1B-Chat-v1.0` doesn't support chat completion via Inference API
- Not all models support all tasks through the API

**Solution:**
- Switched to models that support chat completion: `microsoft/Phi-3-mini-4k-instruct`
- For local hosting, used `HuggingFacePipeline` instead of API endpoint

**Lesson Learned:**
- Check model documentation for supported tasks
- Some models work better locally than via API
- Different models have different capabilities

---

### Issue 5: Missing PyTorch for Local Models

**Problem:**
```
ImportError: AutoModelForCausalLM requires the PyTorch library but it was not found
```

**Root Cause:**
- `HuggingFacePipeline.from_model_id()` requires PyTorch to load models locally
- PyTorch was not installed in virtual environment

**Solution:**
- Installed PyTorch: `pip install torch`
- Verified installation: `python -c "import torch; print(torch.__version__)"`
- Fixed environment variable typo: `HF-HOME` → `HF_HOME`

**Lesson Learned:**
- Local model hosting requires additional dependencies (PyTorch/TensorFlow)
- Environment variable names cannot contain hyphens
- CPU-only PyTorch works but GPU is faster for larger models

---

### Issue 6: Git Security - Preventing API Key Commits

**Problem:**
- Accidentally ran `git add .` before creating `.gitignore`
- Risk of committing `.env` file with API keys

**Root Cause:**
- No `.gitignore` file initially
- Sensitive files not excluded from version control

**Solution:**
- Created comprehensive `.gitignore` file
- Verified no sensitive files were tracked: `git check-ignore .env venv/`
- Documented best practices for Git security

**Lesson Learned:**
- Always create `.gitignore` before first commit
- Use `git status` to verify before committing
- Never commit API keys or sensitive information

---

### Common Issues & Solutions

| Issue                                                | Solution                                            |
| ---------------------------------------------------- | --------------------------------------------------- |
| `ModuleNotFoundError: No module named 'dotenv'`      | Install: `pip install python-dotenv`                |
| `ModuleNotFoundError: No module named 'langchain_*'` | Install provider package: `pip install langchain-*` |
| Python 3.14 compatibility issues                     | Use Python 3.12 instead                             |
| API key not found                                    | Check `.env` file and variable names                |
| `ImportError: PyTorch not found`                     | Install: `pip install torch`                        |
| `StopIteration` with Hugging Face models             | Use chat-completion compatible models               |
| Environment variable not loading                     | Check variable name spelling and `.env` file location |

---

## 🗺️ Future Plans

### Short-term Goals

- [ ] Complete Anthropic Chat Model implementation
- [ ] Add more examples with different parameters
- [ ] Implement error handling and retry logic

### Medium-term Goals

- [ ] Start Embedding Models section
- [ ] Add vector database integration
- [ ] Implement RAG (Retrieval Augmented Generation) examples

### Long-term Goals

- [ ] Build complete applications using LangChain
- [ ] Explore LangGraph for complex workflows
- [ ] Implement agent-based systems

---

## 🔒 Security Notes

- **Never commit API keys or `.env` files**
- The `.gitignore` file is configured to exclude sensitive files
- Always use environment variables for API keys
- Rotate API keys if accidentally exposed

---

## 📖 Resources

- [LangChain Documentation](https://python.langchain.com/)
- [LangChain GitHub](https://github.com/langchain-ai/langchain)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Anthropic API Documentation](https://docs.anthropic.com/)
- [Google AI Studio](https://makersuite.google.com/app/apikey)

---

## 📅 Last Updated

**Date:** January 2025  
**Current Focus:** Chat Models (Local & API)  
**Recent Additions:**
- ✅ Hugging Face local model hosting with PyTorch
- ✅ Comprehensive troubleshooting documentation
- ✅ Multiple Python version management setup

**Next Steps:** Embedding Models

---

## 🤝 Contributing

This is a personal learning repository. Feel free to use the code as reference for your own learning journey!

---

## 📄 License

This project is for educational purposes.

---

**Happy Learning! 🚀**
