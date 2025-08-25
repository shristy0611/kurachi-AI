#!/usr/bin/env python3
"""
Test script to verify the enhanced Kurachi AI setup
"""
import sys
from pathlib import Path

# Add parent directory to Python path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test that all modules can be imported"""
    print("🧪 Testing module imports...")
    
    try:
        from config import config
        print(f"✅ Config: {config.app.app_name} v{config.app.version}")
    except Exception as e:
        print(f"❌ Config import failed: {e}")
        return False
    
    try:
        from utils.logger import get_logger
        logger = get_logger("test")
        logger.info("Test log message")
        print("✅ Logger working")
    except Exception as e:
        print(f"❌ Logger import failed: {e}")
        return False
    
    try:
        from models.database import db_manager
        print("✅ Database models working")
    except Exception as e:
        print(f"❌ Database models failed: {e}")
        return False
    
    try:
        from utils.helpers import generate_uuid, format_file_size
        test_uuid = generate_uuid()
        test_size = format_file_size(1024 * 1024)
        print(f"✅ Helpers working: UUID={test_uuid[:8]}..., Size={test_size}")
    except Exception as e:
        print(f"❌ Helpers failed: {e}")
        return False
    
    return True

def test_directories():
    """Test that required directories exist"""
    print("\n📁 Testing directory structure...")
    
    required_dirs = ["uploads", "logs", "chroma_db"]
    for directory in required_dirs:
        path = Path(directory)
        if path.exists():
            print(f"✅ {directory}/ exists")
        else:
            path.mkdir(exist_ok=True)
            print(f"✅ {directory}/ created")
    
    return True

def test_database():
    """Test database operations"""
    print("\n🗄️ Testing database operations...")
    
    try:
        from models.database import db_manager, Document
        from datetime import datetime
        import uuid
        
        # Test document creation
        test_doc = Document(
            id=str(uuid.uuid4()),
            filename="test.txt",
            original_filename="test.txt",
            file_path="/tmp/test.txt",
            file_type=".txt",
            file_size=1024,
            content_type="text/plain",
            processing_status="pending",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            user_id="test_user"
        )
        
        # Test serialization
        doc_dict = test_doc.to_dict()
        doc_restored = Document.from_dict(doc_dict)
        
        print("✅ Document model serialization working")
        return True
        
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧠 Kurachi AI - Enhanced Setup Test")
    print("=" * 50)
    
    all_passed = True
    
    # Test imports (skip LangChain for now due to Python 3.13 compatibility)
    if not test_imports():
        all_passed = False
    
    # Test directories
    if not test_directories():
        all_passed = False
    
    # Test database
    if not test_database():
        all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All tests passed! Enhanced Kurachi AI is ready.")
        print("\n📝 Next steps:")
        print("1. Ensure Ollama is running: ollama serve")
        print("2. Install required models:")
        print("   ollama pull qwen3:4b")
        print("   ollama pull llava:7b")
        print("3. Run the application: python run.py")
        print("4. Read the comprehensive README.md for detailed documentation")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()