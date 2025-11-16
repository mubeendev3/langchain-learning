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
  - [4. Prompts Component](#4-prompts-component)
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
├── 1.Models Component/        # Model implementations
│   ├── 1.LLMs/                # Basic Language Model implementations
│   │   └── demo_openai.py    # OpenAI LLM demo
│   ├── 2.ChatModels/          # Chat Model implementations
│   │   ├── chat-gemini.py     # Google Gemini Chat Model
│   │   ├── chat-hf-localhosted.py # Hugging Face Chat Model (Local)
│   │   └── chatmodel-huggingface.py # Hugging Face Chat Model (API)
│   └── 3.EmbeddingModels/     # Embedding Model implementations
│       ├── embedding-openai-documents.py # OpenAI embeddings for documents
│       ├── embedding-openai-query.py    # OpenAI embeddings for queries
│       ├── embedding_hf_local_documents.py # Hugging Face local embeddings
│       ├── embedding_hf_local.py         # Hugging Face local query embeddings
│       └── document_similarity.py       # Document similarity using embeddings
├── 2.Prompts Component/       # Prompt engineering and UI
│   ├── prompt_ui_streamlit.py # Streamlit UI for prompt templates
│   ├── chat_prompt_template.py # Chat prompt templates with variables
│   ├── message_placeholder.py # MessagesPlaceholder for chat history
│   ├── messages.py            # SystemMessage, HumanMessage, AIMessage examples
│   ├── prompt_generator.py    # Generate and save prompt templates
│   ├── prompt_template.json  # Saved prompt template (JSON format)
│   └── chat_history.txt       # Example chat history file
├── Projects/                  # Complete project implementations
│   └── 1.Chatbot/            # Chatbot project
│       └── chatbot.py        # Interactive chatbot with chat history
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
   git clone 'https://github.com/mubeendev3/langchain-learning.git'
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

### 0. Projects

**Status:** ✅ In Progress

Complete application implementations combining multiple LangChain components.

#### Implementations:

1. **Interactive Chatbot** (`Projects/1.Chatbot/chatbot.py`)
   - Full-featured chatbot with persistent chat history
   - Features:
     - System message configuration
     - Continuous conversation loop
     - Chat history management (appends user and AI messages)
     - Exit command support
     - Model: `gpt-4o-mini`
   - Status: ✅ Working
   - Key Features:
     - Uses `SystemMessage`, `HumanMessage`, and `AIMessage`
     - Maintains conversation context across multiple turns
     - Simple command-line interface

**Key Learnings:**

- Building complete applications with LangChain
- Managing conversation state and history
- Combining multiple LangChain components (models, messages, prompts)
- Creating interactive user experiences

---

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

**Status:** ✅ Completed

Embedding models for converting text into vector representations for semantic search and similarity matching.

#### Implementations:

1. **OpenAI Embeddings**

   - `embedding-openai-documents.py` - Embed multiple documents
   - `embedding-openai-query.py` - Embed single query
   - Model: `text-embedding-3-small`
   - Features: Configurable dimensions (32, 300, etc.)
   - Status: ✅ Working

2. **Hugging Face Local Embeddings**

   - `embedding_hf_local_documents.py` - Embed documents locally
   - `embedding_hf_local.py` - Embed queries locally
   - Model: `sentence-transformers/all-MiniLM-L6-v2`
   - Features: Runs locally, no API required
   - Status: ✅ Working

3. **Document Similarity** (`document_similarity.py`)
   - Uses OpenAI embeddings
   - Calculates cosine similarity between query and documents
   - Finds most similar document to a query
   - Status: ✅ Working

**Key Learnings:**

- Embeddings convert text to numerical vectors
- Cosine similarity measures semantic similarity
- Different embedding models have different dimensions
- Local embeddings don't require API keys
- Embeddings enable semantic search and RAG applications

---

### 4. Prompts Component

**Status:** ✅ Completed

Comprehensive prompt engineering examples covering templates, chat history, message types, and interactive UIs.

#### Implementations:

1. **Streamlit Prompt UI** (`2.Prompts Component/prompt_ui_streamlit.py`)

   - Interactive web interface for prompt templates
   - Features:
     - Research paper selection dropdown
     - Explanation style selection (Beginner-Friendly, Technical, Code-Oriented, Mathematical)
     - Explanation length selection (Short, Medium, Long)
     - Dynamic prompt template with user inputs
     - Loads templates from JSON files using `load_prompt()`
   - Model: `gpt-4o-mini`
   - Status: ✅ Working

2. **Chat Prompt Templates** (`2.Prompts Component/chat_prompt_template.py`)

   - Demonstrates `ChatPromptTemplate.from_messages()` method
   - Shows how to use variables in chat prompts (e.g., `{domain}`, `{topic}`)
   - Creates structured prompts with system and human messages
   - Status: ✅ Working

3. **Message Placeholder** (`2.Prompts Component/message_placeholder.py`)

   - Shows how to use `MessagesPlaceholder` for chat history
   - Demonstrates loading chat history from files
   - Integrates chat history into prompt templates
   - Status: ✅ Working

4. **Message Types** (`2.Prompts Component/messages.py`)

   - Examples of different message types:
     - `SystemMessage` - System instructions
     - `HumanMessage` - User inputs
     - `AIMessage` - AI responses
   - Shows how to build conversation history
   - Demonstrates appending messages to maintain context
   - Status: ✅ Working

5. **Prompt Generator** (`2.Prompts Component/prompt_generator.py`)
   - Creates `PromptTemplate` with multiple input variables
   - Saves templates to JSON format using `template.save()`
   - Demonstrates template persistence and reuse
   - Status: ✅ Working

**Key Learnings:**

- `PromptTemplate` allows dynamic prompt creation with variables
- `ChatPromptTemplate` is specifically designed for chat-based interactions
- `MessagesPlaceholder` enables dynamic chat history insertion
- Different message types (`SystemMessage`, `HumanMessage`, `AIMessage`) structure conversations
- Templates can be saved to JSON and loaded later with `load_prompt()`
- Streamlit provides easy web UI for LangChain applications
- Template variables can be filled from user inputs
- Structured prompts improve AI response quality
- Chat history management is crucial for conversational AI
- Message-based architecture allows for better conversation flow control

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

**For Embeddings & Similarity:**

- `scikit-learn` - For cosine similarity calculations
- `numpy` - For numerical operations

**For UI Components:**

- `streamlit` - For building interactive web interfaces

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
python .\1.Models Component\2.ChatModels\chat-hf-localhosted.py

# Embedding Models
python .\1.Models Component\3.EmbeddingModels\embedding-openai-documents.py
python .\1.Models Component\3.EmbeddingModels\embedding-openai-query.py
python .\1.Models Component\3.EmbeddingModels\document_similarity.py

# Prompts Component (Streamlit UI)
streamlit run .\2.Prompts Component\prompt_ui_streamlit.py

# Prompt Component Examples
python .\2.Prompts Component\chat_prompt_template.py
python .\2.Prompts Component\message_placeholder.py
python .\2.Prompts Component\messages.py
python .\2.Prompts Component\prompt_generator.py

# Projects
python .\Projects\1.Chatbot\chatbot.py
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

6. **Prompt Templates & Messages**

   - `PromptTemplate` is for simple text prompts with variables
   - `ChatPromptTemplate` is for structured chat conversations
   - `MessagesPlaceholder` allows dynamic insertion of chat history
   - Message types (`SystemMessage`, `HumanMessage`, `AIMessage`) structure conversations
   - Templates can be saved to JSON and loaded with `load_prompt()`
   - Chat history should be maintained as a list of message objects

7. **Chat History Management**
   - Append both user messages and AI responses to maintain context
   - Use `AIMessage(content=result.content)` to store AI responses
   - Chat history can be loaded from files or maintained in memory
   - System messages should typically be at the start of the conversation

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

| Issue                                                | Solution                                              |
| ---------------------------------------------------- | ----------------------------------------------------- |
| `ModuleNotFoundError: No module named 'dotenv'`      | Install: `pip install python-dotenv`                  |
| `ModuleNotFoundError: No module named 'langchain_*'` | Install provider package: `pip install langchain-*`   |
| Python 3.14 compatibility issues                     | Use Python 3.12 instead                               |
| API key not found                                    | Check `.env` file and variable names                  |
| `ImportError: PyTorch not found`                     | Install: `pip install torch`                          |
| `StopIteration` with Hugging Face models             | Use chat-completion compatible models                 |
| Environment variable not loading                     | Check variable name spelling and `.env` file location |

---

## 🗺️ Future Plans

### Short-term Goals

- [x] Complete Prompts Component with all message types ✅
- [x] Build interactive chatbot project ✅
- [ ] Complete Anthropic Chat Model implementation
- [ ] Add more Streamlit UI examples
- [ ] Implement error handling and retry logic
- [ ] Add vector database integration examples

### Medium-term Goals

- [x] Complete Embedding Models section ✅
- [ ] Add vector database integration (Pinecone, Weaviate, Chroma)
- [ ] Implement RAG (Retrieval Augmented Generation) examples
- [ ] Build more interactive UIs with Streamlit

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
**Current Focus:** Prompts Component & Projects  
**Recent Additions:**

- ✅ Complete Prompts Component implementation:
  - Chat prompt templates with variables
  - MessagesPlaceholder for chat history
  - SystemMessage, HumanMessage, AIMessage examples
  - Prompt template generation and JSON persistence
  - Updated Streamlit UI with template loading
- ✅ Interactive Chatbot project with chat history management
- ✅ Chat history file handling and persistence
- ✅ Template saving and loading functionality
- ✅ Embedding Models section completed (OpenAI & Hugging Face)
- ✅ Document similarity implementation with cosine similarity
- ✅ Project structure reorganization (Models Component, Prompts Component, Projects)
- ✅ Hugging Face local model hosting with PyTorch
- ✅ Comprehensive troubleshooting documentation

**Next Steps:**

- Vector databases integration
- RAG (Retrieval Augmented Generation) implementation
- Chains and Agents
- More complete project implementations

---

## 🤝 Contributing

This is a personal learning repository. Feel free to use the code as reference for your own learning journey!

---

## 📄 License

This project is for educational purposes.

---

**Happy Learning! 🚀**
