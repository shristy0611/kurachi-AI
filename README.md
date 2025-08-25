# 🧠 Kurachi AI (蔵地AI) - Enhanced Business Platform

**Kurachi AI** is a 100% local document intelligence platform with business-grade features, using Ollama models with advanced RAG capabilities. Designed for Japanese businesses that need complete data privacy and professional document management.

## 🚀 Enhanced Features

- **🏗️ Business-Grade Architecture** - Modular, scalable, and maintainable codebase
- **📄 Advanced Document Management** - Upload, process, and manage documents with status tracking
- **💬 Conversation Management** - Persistent chat history with multiple conversations
- **📊 Analytics Dashboard** - Usage metrics and document processing insights
- **🔒 Security & Audit** - User sessions, audit logging, and compliance features
- **🌐 Enhanced UI** - Professional Streamlit interface with file upload and management
- **100% Local AI** - No internet required, complete data privacy
- **Japanese-English Bilingual** - Optimized for Japanese business documents
- **Universal Document Processing** - PDFs, Word, Excel, PowerPoint, images, and more

## 🛠️ Quick Start

### Prerequisites
1. **Python 3.11 or 3.12** (recommended for LangChain compatibility)
2. **Ollama** with required models

### Installation
```bash
# 1. Install Ollama and models
ollama serve
ollama pull qwen3:4b
ollama pull llava:7b

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up environment
cp .env.example .env
# Edit .env as needed

# 4. Run the application
python run.py
```

### Access the Application
- Open your browser to `http://localhost:8501`
- Upload documents, start conversations, and explore analytics!

## 🏗️ Architecture Overview

### Directory Structure
```
kurachi-ai/
├── 📄 Configuration & Entry Points
│   ├── config.py                 # Environment-based configuration management
│   ├── main.py                   # Main application entry point
│   ├── app.py                    # Legacy compatibility entry point
│   ├── run.py                    # Convenient startup script with checks
│   ├── .env / .env.example       # Environment variables
│   └── requirements.txt          # Python dependencies
│
├── 🗄️ Data Models
│   └── models/
│       └── database.py           # SQLite models (Document, Session, Chat)
│
├── ⚙️ Business Services
│   └── services/
│       ├── document_service.py   # Document processing & management
│       └── chat_service.py       # RAG chat & conversation management
│
├── 🖥️ User Interface
│   └── ui/
│       └── streamlit_app.py      # Enhanced Streamlit web interface
│
├── 🛠️ Utilities
│   └── utils/
│       ├── logger.py             # Structured logging & audit system
│       ├── exceptions.py         # Custom exception classes
│       └── helpers.py            # Common utility functions
│
├── 📂 Data Directories
│   ├── uploads/                  # User uploaded documents
│   ├── chroma_db/               # Vector database (ChromaDB)
│   ├── logs/                    # Application & audit logs
│   ├── data/                    # Application data (SQLite database)
│   └── documents/               # Sample/demo documents
│
└── 🧪 Testing
    └── tests/
        └── test_setup.py         # Setup verification script
```

### Key Components

#### Configuration (`config.py`)
- Environment-based settings for different deployment scenarios
- AI model configuration (LLM, vision, embedding models)
- Database paths and security settings
- Application metadata and feature flags

#### Database Models (`models/database.py`)
- **Document**: File metadata, processing status, user association
- **UserSession**: Session management and tracking
- **ChatConversation**: Conversation metadata and organization
- **ChatMessage**: Individual messages with metadata

#### Services
- **Document Service**: Upload, process, manage documents with status tracking
- **Chat Service**: RAG conversations with context and conversation history

#### User Interface (`ui/streamlit_app.py`)
- Multi-page Streamlit application (Chat, Documents, Analytics)
- Document management interface with drag-and-drop upload
- Chat interface with conversation history and source citations
- Analytics dashboard with usage metrics

## 📱 User Interface Features

### 💬 Chat Interface
- **Multi-conversation support**: Create and manage multiple chat sessions
- **Document-aware responses**: AI responses include source citations
- **Conversation history**: Persistent chat history across sessions
- **Context-aware**: Uses conversation history for better responses

### 📄 Document Management
- **File upload**: Drag-and-drop or browse to upload documents
- **Processing status**: Track document processing in real-time
- **Document library**: View and manage all uploaded documents
- **File type support**: PDF, Word, Excel, PowerPoint, images, and more

### 📊 Analytics Dashboard
- **Usage statistics**: Document and chat metrics
- **Processing insights**: Success rates and error tracking
- **User activity**: Conversation and document trends

## 🔧 Configuration

The application uses environment variables for configuration. Key settings:

```bash
# Environment
KURACHI_ENV=development                    # development, staging, production
KURACHI_DEBUG=true
KURACHI_LOG_LEVEL=INFO

# AI Models
KURACHI_LLM_MODEL=qwen3:4b
KURACHI_VISION_MODEL=llava:7b
KURACHI_EMBEDDING_MODEL=qwen3:4b
KURACHI_OLLAMA_URL=http://localhost:11434

# Application
KURACHI_MAX_FILE_SIZE_MB=100
KURACHI_UPLOAD_DIR=./uploads

# Database
KURACHI_DB_PATH=data/kurachi.db
KURACHI_CHROMA_DIR=./chroma_db

# Security
KURACHI_SECRET_KEY=your-secret-key
KURACHI_SESSION_TIMEOUT_HOURS=24
KURACHI_ENABLE_AUDIT_LOGGING=true
```

## 🗄️ Database Schema

The application uses SQLite for metadata and ChromaDB for vectors:

- **Documents**: File metadata, processing status, user association
- **User Sessions**: Session management and tracking
- **Chat Conversations**: Conversation metadata and organization
- **Chat Messages**: Individual messages with metadata
- **Vector Store**: Document embeddings (ChromaDB)

## 🔍 API Structure

### Document Service
```python
# Upload and process documents
document = document_service.save_uploaded_file(file, user_id)
success = document_service.process_document(document.id)

# Search documents
results = document_service.search_documents(query, user_id)

# Get user documents
documents = document_service.get_user_documents(user_id)
```

### Chat Service
```python
# Create conversation
conversation = chat_service.create_conversation(user_id, title)

# Send message
response = chat_service.send_message(conversation_id, message, user_id)

# Stream response
for chunk in chat_service.stream_response(conversation_id, message, user_id):
    print(chunk)

# Get conversation history
messages = chat_service.get_conversation_history(conversation_id, user_id)
```

## 📝 Logging & Monitoring

- **Structured logging**: JSON-formatted logs with context
- **Audit trails**: User actions and system events
- **Error tracking**: Comprehensive error logging with stack traces
- **Performance monitoring**: Operation timing and metrics

## 🔒 Security Features

- **Session management**: Secure user sessions with expiration
- **Audit logging**: Complete audit trail for compliance
- **Input validation**: File type and size validation
- **Error handling**: Secure error messages without information leakage
- **Local processing**: No external API calls, complete data privacy

## 🚀 Entry Points

| File | Purpose | Usage |
|------|---------|-------|
| `main.py` | Primary entry point | `python main.py` |
| `run.py` | Startup script with checks | `python run.py` |
| `app.py` | Legacy compatibility | `streamlit run app.py` |

## 🔄 Data Flow

1. **Document Upload** → Document Service → Database + Vector Store
2. **Chat Query** → Chat Service → RAG Processing → Response
3. **User Actions** → Audit Logger → Audit Logs
4. **Configuration** → Environment Variables → Application Settings

## 🐍 Python Compatibility

### Recommended: Python 3.11 or 3.12
The enhanced Kurachi AI architecture works best with Python 3.11 or 3.12 due to LangChain dependencies.

### Python 3.13 Compatibility Issue
There's a compatibility issue with Python 3.13 and current LangChain dependencies (the `cgi` module was removed).

#### Solutions:
1. **Use Python 3.11/3.12** (Recommended):
   ```bash
   # Using pyenv
   pyenv install 3.11.9
   pyenv local 3.11.9
   
   # Or using conda
   conda create -n kurachi python=3.11
   conda activate kurachi
   pip install -r requirements.txt
   ```

2. **Wait for LangChain Python 3.13 support**
3. **Use alternative dependencies** (replace LangChain components)

### Testing Your Setup
Run the setup verification script:
```bash
python test_setup.py
```

This will confirm that all the enhanced components are working correctly.

## 🔄 Migration from Legacy Version

The enhanced version maintains backward compatibility:

1. **Existing ChromaDB**: Your existing vector database will be automatically loaded
2. **Document processing**: Improved processing with metadata tracking
3. **Chat interface**: Enhanced UI with conversation management
4. **Configuration**: New environment-based configuration system

## 🎯 What's New

This enhanced version implements **Task 1** from the Kurachi AI specification:

- ✅ **Modular Architecture**: Clean separation with services, models, utilities
- ✅ **Configuration Management**: Environment-based configuration
- ✅ **Database Models**: Document metadata, user sessions, chat conversations
- ✅ **Logging & Audit**: Comprehensive logging with audit trails
- ✅ **Error Handling**: Structured error handling throughout
- ✅ **Enhanced UI**: Streamlit app with document management and analytics
- ✅ **Business Features**: User sessions, audit logging, security features

### Recent Optimizations

The codebase has been optimized for better maintainability and performance:

- ✅ **Consolidated Test Runners**: Removed duplicate test runners, keeping only the most optimized versions
- ✅ **Standardized Naming**: Updated file names to follow modern Python conventions
- ✅ **Cleaned Analysis Files**: Removed redundant analysis files while preserving comprehensive summaries
- ✅ **Improved Performance**: Test execution optimized for faster development cycles

## 🛣️ Roadmap

**Next phases** will include:
- Advanced document processing (Task 2)
- Knowledge graph construction (Task 3)
- Bilingual translation system (Task 5)
- Analytics and insights (Task 6)

## 🧪 Testing & Development

### Setup Verification
```bash
python tests/test_setup.py
```

### Test Runners
The project includes optimized test runners for efficient development:

```bash
# Fast test runner (optimized for development)
python test_runner_fast.py

# Comprehensive test runner (full test suite)
python test_runner_comprehensive.py

# Standard pytest execution
python -m pytest tests/
```

### Running the Application
```bash
# Development mode
python run.py

# Direct execution
python main.py

# Legacy compatibility
streamlit run app.py
```

## 🤝 Contributing

This is a business-grade enhancement of the original RAG chatbot. The modular architecture makes it easy to extend and customize for specific business needs.

## 📄 License

This enhanced version builds upon the original codebase and maintains the same licensing terms.

---

**🧠 Kurachi AI** - Professional document intelligence for Japanese businesses with complete local privacy and advanced AI capabilities.