"""
Legacy app.py - Redirects to new modular architecture
This file maintains backward compatibility while using the new enhanced system
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import and run the new Streamlit app
from ui.streamlit_app import main

if __name__ == "__main__":
    print("🧠 Starting Kurachi AI - Enhanced Business RAG Platform")
    print("Note: This is the legacy entry point. Use 'python main.py' for the full experience.")
    main()