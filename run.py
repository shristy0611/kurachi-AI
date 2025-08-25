#!/usr/bin/env python3
"""
Kurachi AI - SOTA Startup Script
Convenient script to run the application with SOTA enhancements and proper setup
"""
import sys
import os
import subprocess
import time
from pathlib import Path
from services.sota_config import sota_config, get_config, validate_config
from services.sota_observability import record_metric, MetricType

def check_requirements():
    """Check if required packages are installed"""
    try:
        import streamlit
        import langchain
        import chromadb
        print("✅ All required packages are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        return False

def check_ollama():
    """Check if Ollama is running and models are available using SOTA config"""
    try:
        config = get_config()
        ai_config = getattr(config, 'ai', None)
        
        # Get configuration values with fallbacks
        if ai_config:
            ollama_url = getattr(ai_config, 'ollama_base_url', 'http://localhost:11434')
            llm_model = getattr(ai_config, 'llm_model', 'qwen3:4b')
            embedding_model = getattr(ai_config, 'embedding_model', 'nomic-embed-text')
        else:
            # Fallback values
            ollama_url = 'http://localhost:11434'
            llm_model = 'qwen3:4b'
            embedding_model = 'nomic-embed-text'
        
        import requests
        response = requests.get(f"{ollama_url}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [model["name"] for model in models]
            
            # Verify required models
            required_models = [llm_model, embedding_model]
            missing_models = [model for model in required_models if model not in model_names]
            
            if missing_models:
                print(f"❌ Missing Ollama models: {', '.join(missing_models)}")
                print("Please install missing models:")
                for model in missing_models:
                    print(f"  ollama pull {model}")
                record_metric("startup_check_failure", 1, MetricType.COUNTER, {"reason": "missing_models"})
                return False
            else:
                print("✅ Ollama is running with required models")
                record_metric("startup_check_success", 1, MetricType.COUNTER, {"component": "ollama"})
                return True
        else:
            print("❌ Ollama is not responding properly")
            record_metric("startup_check_failure", 1, MetricType.COUNTER, {"reason": "ollama_not_responding"})
            return False
    except Exception as e:
        print(f"❌ Cannot connect to Ollama: {e}")
        print("Please ensure Ollama is running: ollama serve")
        record_metric("startup_check_failure", 1, MetricType.COUNTER, {"reason": "connection_error"})
        return False

def setup_environment():
    """Set up environment variables and directories"""
    # Create .env file if it doesn't exist
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if not env_file.exists() and env_example.exists():
        print("📝 Creating .env file from template...")
        env_file.write_text(env_example.read_text())
        print("✅ .env file created. You can customize it as needed.")
    
    # Create necessary directories
    directories = ["uploads", "logs", "chroma_db", "data", "tests"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    
    print("✅ Environment setup complete")

def run_streamlit():
    """Run the Streamlit application"""
    print("🚀 Starting Kurachi AI...")
    
    # Set environment variables for Streamlit
    os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
    os.environ["STREAMLIT_SERVER_PORT"] = "8501"
    # Relax protections for local development when running behind proxies
    # This helps avoid 403s from the Streamlit file uploader (Axios) due to XSRF/CORS
    os.environ.setdefault("STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION", "false")
    os.environ.setdefault("STREAMLIT_SERVER_ENABLE_CORS", "false")
    
    # Run Streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "main.py",
            "--server.headless", "true",
            "--server.port", "8501",
            "--server.address", "0.0.0.0",
            "--server.enableXsrfProtection", os.environ.get("STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION", "false"),
            "--server.enableCORS", os.environ.get("STREAMLIT_SERVER_ENABLE_CORS", "false"),
        ], check=True)
    except KeyboardInterrupt:
        print("\n👋 Kurachi AI stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start Streamlit: {e}")

def main():
    """Main startup function with SOTA enhancements"""
    startup_start = time.time()
    
    print("🧠 Kurachi AI - Business RAG Platform (SOTA Enhanced)")
    print("=" * 60)
    
    # Get SOTA configuration
    config = get_config()
    
    # Validate configuration
    print("📝 Validating SOTA configuration...")
    config_issues = validate_config()
    if config_issues:
        print(f"⚠️ Configuration issues found: {len(config_issues)}")
        for issue in config_issues:
            print(f"  - {issue}")
    else:
        print("✅ SOTA configuration validated successfully")
    
    # Check system requirements
    print("🔍 Checking system requirements...")
    if not check_requirements():
        record_metric("startup_failure", 1, MetricType.COUNTER, {"reason": "requirements"})
        sys.exit(1)
    
    # Check Ollama with SOTA config
    print("🤖 Checking Ollama with SOTA configuration...")
    if not check_ollama():
        config = get_config()
        ai_config = getattr(config, 'ai', None)
        
        if ai_config:
            llm_model = getattr(ai_config, 'llm_model', 'qwen3:4b')
            embedding_model = getattr(ai_config, 'embedding_model', 'nomic-embed-text')
        else:
            llm_model = 'qwen3:4b'
            embedding_model = 'nomic-embed-text'
        
        print("\n💡 To install Ollama and models:")
        print("1. Install Ollama: https://ollama.ai/")
        print("2. Start Ollama: ollama serve")
        print("3. Install models:")
        print(f"   ollama pull {llm_model}")
        print(f"   ollama pull {embedding_model}")
        record_metric("startup_failure", 1, MetricType.COUNTER, {"reason": "ollama"})
        sys.exit(1)
    
    # Setup environment
    print("🛠️ Setting up environment...")
    setup_environment()
    
    # Record startup performance
    startup_time = (time.time() - startup_start) * 1000
    record_metric("startup_preparation_time_ms", startup_time, MetricType.HISTOGRAM)
    
    print("\n🎯 All checks passed! Starting SOTA-enhanced application...")
    print(f"📊 Startup preparation completed in {startup_time:.2f}ms")
    print("📱 Access the app at: http://localhost:8501")
    print("🛑 Press Ctrl+C to stop")
    print("-" * 60)
    
    # Run the application
    run_streamlit()

if __name__ == "__main__":
    main()