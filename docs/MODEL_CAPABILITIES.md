# Model Capabilities and Dependencies

This document clarifies what AI models and capabilities are available in the Kurachi AI document processing system.

## Local Ollama Models (Running Locally)

These models run on your local Ollama instance and are the core of the Kurachi AI system:

### 🤖 **qwen3:4b** (LLM Model)
- **Purpose**: Text generation, chat responses, document analysis
- **Capabilities**: 
  - Natural language understanding and generation
  - Document summarization
  - Question answering
  - Text analysis
- **Location**: Local Ollama server
- **Configuration**: `config.ai.llm_model`

### 👁️ **llava:7b** (Vision Model) 
- **Purpose**: Image understanding and analysis
- **Capabilities**:
  - Image content description
  - Visual element identification
  - Chart and diagram analysis
  - Document structure analysis (for task 2.2)
- **Location**: Local Ollama server
- **Configuration**: `config.ai.vision_model`
- **Note**: This will be used in task 2.2 for document structure analysis

### 🔍 **qwen3:4b** (Embedding Model)
- **Purpose**: Text embeddings for vector search
- **Capabilities**:
  - Convert text to vector representations
  - Enable semantic search
  - Document similarity matching
- **Location**: Local Ollama server
- **Configuration**: `config.ai.embedding_model`

## External Models (Not Local Ollama)

These are external dependencies that provide specialized capabilities:

### 🎤 **OpenAI Whisper** (Speech-to-Text)
- **Purpose**: Audio and video transcription
- **Capabilities**:
  - Speech-to-text conversion
  - Multi-language support
  - Audio extraction from video
- **Location**: External Python library (runs locally but not via Ollama)
- **Installation**: `pip install whisper`
- **Configuration**: `config.ai.whisper_model` (base/small/medium/large)
- **Status**: Optional - graceful fallback if not available

### 👀 **Tesseract OCR** (Optical Character Recognition)
- **Purpose**: Text extraction from images and scanned documents
- **Capabilities**:
  - Extract text from images
  - Multi-language OCR (English + Japanese)
  - Scanned document processing
- **Location**: External system package + Python wrapper
- **Installation**: System package + `pip install pytesseract`
- **Configuration**: `config.ai.ocr_language`
- **Status**: Optional - graceful fallback if not available

### 🎬 **FFmpeg** (Video Processing)
- **Purpose**: Audio extraction from video files
- **Capabilities**:
  - Extract audio tracks from video
  - Format conversion
  - Video metadata extraction
- **Location**: External system package
- **Installation**: System package (brew install ffmpeg on macOS)
- **Status**: Required for video processing - graceful fallback if not available

## Processing Capabilities by File Type

| File Type | Local Model Used | External Tool Used | Capability |
|-----------|------------------|-------------------|------------|
| **PDF** | qwen3:4b (embeddings) | Tesseract OCR (fallback) | Text extraction + OCR |
| **Images** | llava:7b (analysis) | Tesseract OCR | Visual analysis + text extraction |
| **Audio** | qwen3:4b (embeddings) | Whisper | Speech-to-text transcription |
| **Video** | qwen3:4b (embeddings) | FFmpeg + Whisper | Audio extraction + transcription |
| **Text/Code** | qwen3:4b (embeddings) | None | Direct text processing |
| **Office Docs** | qwen3:4b (embeddings) | None | Structured content extraction |

## Configuration Options

### Enable/Disable External Capabilities

```python
# In config.py
@dataclass
class AppConfig:
    enable_ocr: bool = True              # Enable Tesseract OCR
    enable_audio_transcription: bool = True  # Enable Whisper
    enable_video_processing: bool = True     # Enable FFmpeg + Whisper
```

### Model Selection

```python
# In config.py
@dataclass 
class AIConfig:
    # Local Ollama models
    llm_model: str = "qwen3:4b"
    vision_model: str = "llava:7b" 
    embedding_model: str = "qwen3:4b"
    
    # External models
    whisper_model: str = "base"  # base/small/medium/large
    ocr_language: str = "eng+jpn"  # Tesseract language codes
```

## Graceful Fallbacks

The system is designed to work even when external dependencies are not available:

1. **No Tesseract**: PDF/image processing falls back to basic text extraction
2. **No Whisper**: Audio/video files are indexed with metadata only
3. **No FFmpeg**: Video files are indexed with metadata only
4. **No llava:7b**: Image analysis falls back to OCR-only processing

## Installation Requirements

### Minimal Setup (Local models only)
```bash
# Only requires Ollama and basic Python packages
pip install langchain langchain-community chromadb unstructured
```

### Full Setup (All capabilities)
```bash
# System packages
brew install tesseract ffmpeg  # macOS
# or
apt-get install tesseract-ocr ffmpeg  # Ubuntu

# Python packages
pip install whisper pytesseract pillow opencv-python pymupdf
```

## Future Enhancements

### Potential Local Model Integrations
- **Audio transcription**: Could potentially use a local speech-to-text model via Ollama
- **OCR**: Could potentially use a local OCR model via Ollama
- **Video analysis**: Could use llava:7b for video frame analysis

### Current Limitations
- Audio/video transcription requires external Whisper model
- OCR requires external Tesseract
- No local speech-to-text model available via Ollama yet

This architecture provides maximum capability when all dependencies are available, while maintaining core functionality with just the local Ollama models.