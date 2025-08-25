# Optional Features Setup

This document describes how to enable optional features in Kurachi AI. The core system works without these, but they provide enhanced functionality.

## 🧠 spaCy Language Models

spaCy models are used for advanced text processing and entity extraction.

### Installation

```bash
# Install spaCy models (choose based on your needs)
python -m spacy download en_core_web_sm    # English (small, fast)
python -m spacy download en_core_web_md    # English (medium, more accurate)
python -m spacy download ja_core_news_sm   # Japanese (small)
python -m spacy download ja_core_news_md   # Japanese (medium)
```

### Environment Variables

```bash
# Optional: Disable spaCy if not needed
export ENABLE_SPACY=0
```

### Verification

```bash
python -c "import spacy; print('spaCy models:', spacy.util.get_installed_models())"
```

## 🗄️ Neo4j Knowledge Graph

Neo4j provides advanced knowledge graph capabilities for document relationships.

### Docker Setup (Recommended)

```bash
# Create docker-compose.yml
cat > docker-compose.yml << EOF
version: '3.8'
services:
  neo4j:
    image: neo4j:5.15
    ports:
      - "7474:7474"  # HTTP
      - "7687:7687"  # Bolt
    environment:
      - NEO4J_AUTH=neo4j/password
      - NEO4J_PLUGINS=["apoc"]
    volumes:
      - neo4j_data:/data
      - neo4j_logs:/logs
volumes:
  neo4j_data:
  neo4j_logs:
EOF

# Start Neo4j
docker-compose up -d neo4j
```

### Environment Variables

```bash
# Enable Neo4j
export ENABLE_NEO4J=1
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USERNAME=neo4j
export NEO4J_PASSWORD=password
```

### Verification

```bash
# Check if Neo4j is running
curl http://localhost:7474/
```

## 📄 PDF Processing with OCR

Enhanced PDF processing with OCR capabilities using Unstructured.

### Installation

```bash
# Install PDF processing extras
pip install "unstructured[pdf]"

# For advanced OCR (optional)
pip install "unstructured[all-docs]"
```

### System Dependencies

**macOS:**
```bash
brew install tesseract poppler
```

**Ubuntu/Debian:**
```bash
sudo apt-get install tesseract-ocr poppler-utils
```

### Environment Variables

```bash
# Configure OCR languages
export OCR_LANGUAGES=eng+jpn  # English + Japanese
export ENABLE_OCR=1
```

### Verification

```bash
python -c "
try:
    from unstructured.partition.pdf import partition_pdf
    print('✅ PDF processing available')
except ImportError as e:
    print('❌ PDF processing not available:', e)
"
```

## 🎵 Audio/Video Processing

Whisper-based transcription for audio and video files.

### Installation

```bash
# Install Whisper
pip install openai-whisper

# For faster inference (optional)
pip install whisper-cpp-python
```

### Environment Variables

```bash
# Configure Whisper model size
export WHISPER_MODEL=base  # tiny, base, small, medium, large
export ENABLE_AUDIO_TRANSCRIPTION=1
export ENABLE_VIDEO_PROCESSING=1
```

### Verification

```bash
python -c "
try:
    import whisper
    print('✅ Whisper available')
    print('Models:', whisper.available_models())
except ImportError:
    print('❌ Whisper not available')
"
```

## 🔧 Development Tools

### Validation and Testing

```bash
# Run validation gate (CI-ready)
python scripts/ci_validation_gate.py

# Clean up old backups
python scripts/cleanup_backups.py --dry-run
python scripts/cleanup_backups.py --keep 3

# Run specific validation tests
python tools/validation_system.py --imports-only
python tools/validation_system.py --functionality-only
```

### Performance Monitoring

```bash
# Capture performance baseline
python tools/execute_validation.py --mode pre

# Run post-optimization validation
python tools/execute_validation.py --mode post
```

## 🚀 CI/CD Integration

### GitHub Actions Example

```yaml
name: Validation Gate
on: [push, pull_request]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -r requirements-lock.txt
          
      - name: Run validation gate
        run: python scripts/ci_validation_gate.py
        
      - name: Clean up artifacts
        run: python scripts/cleanup_backups.py --keep 1
```

## 📊 Monitoring and Metrics

### Environment Variables for Monitoring

```bash
# Enable detailed logging
export LOG_LEVEL=DEBUG
export ENABLE_AUDIT_LOGGING=1

# Performance monitoring
export ENABLE_PERFORMANCE_MONITORING=1
export METRICS_COLLECTION=1
```

### Health Checks

```bash
# Quick health check
python -c "
from tools.validation_system import ValidationSystem
validator = ValidationSystem()
result = validator.validate_imports()
print(f'System health: {result[\"passed\"]}/{result[\"passed\"] + result[\"failed\"]} services OK')
"
```

## 🛠️ Troubleshooting

### Common Issues

1. **spaCy models not found**
   ```bash
   python -m spacy download en_core_web_sm
   ```

2. **Neo4j connection refused**
   ```bash
   docker-compose up -d neo4j
   # Wait 30 seconds for startup
   ```

3. **PDF processing fails**
   ```bash
   pip install "unstructured[pdf]"
   brew install tesseract poppler  # macOS
   ```

4. **Validation tests timeout**
   ```bash
   export VALIDATION_TIMEOUT=300  # 5 minutes
   ```

### Getting Help

- Check logs in `logs/` directory
- Run validation with `--debug` flag
- Use `python tools/test_validation_system.py` for self-diagnostics