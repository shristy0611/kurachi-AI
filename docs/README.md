# Kurachi AI - Multilingual Knowledge Platform

## 🏗️ Architecture Overview

Kurachi AI is a state-of-the-art multilingual knowledge platform that provides intelligent document processing, cross-language search, and culturally-adapted responses for Japanese business environments.

## 📁 Project Structure

```
kurachi-ai/
├── 📚 docs/                          # Documentation
│   ├── architecture/                 # System architecture docs
│   ├── implementation/               # Implementation guides
│   └── testing/                      # Testing documentation
├── 🧪 tests/                         # Test suites
│   ├── unit/                         # Unit tests
│   ├── integration/                  # Integration tests
│   └── smoke/                        # End-to-end smoke tests
├── 🛠️ tools/                         # CLI tools and scripts
│   ├── cli/                          # Command-line interfaces
│   └── scripts/                      # Utility scripts
├── 🔧 services/                      # Core business logic
├── 🎨 ui/                            # User interface components
├── 📊 models/                        # Data models and database
├── ⚙️ config/                        # Configuration files
├── 🔌 utils/                         # Utility functions
└── 📦 data/                          # Data storage
```

## 🚀 Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**
   ```bash
   python run.py
   ```

3. **Access Web Interface**
   ```
   http://localhost:8501
   ```

## 🌟 Key Features

- **Multilingual Document Processing** - Japanese/English document ingestion and analysis
- **Cross-Language Search** - Query in one language, search across all languages
- **Cultural Adaptation** - Business-appropriate responses with Japanese keigo
- **Intelligent Translation** - Context-aware translation with business glossaries
- **Real-time Analytics** - Performance monitoring and usage metrics

## 📖 Documentation

- [Architecture Guide](architecture/) - System design and components
- [Implementation Guide](implementation/) - Development and deployment
- [Testing Guide](testing/) - Test strategies and execution

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/

# Fast test runner (optimized for development)
python test_runner_fast.py

# Comprehensive test runner (full test suite)
python test_runner_comprehensive.py

# Run smoke tests
python tests/smoke/test_smoke_multilingual_pipeline.py

# Run specific test suite
python tests/unit/test_preference_enum_serialization.py
```

## 🛠️ CLI Tools

```bash
# Manage user preferences
python tools/cli/preferences.py view user123

# Run system diagnostics
python tools/scripts/system_diagnostics.py
```

## 📊 Monitoring

- **Metrics Dashboard** - Real-time performance metrics
- **Health Checks** - System status monitoring
- **Audit Logs** - User action tracking

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.