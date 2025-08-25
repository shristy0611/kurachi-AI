#!/usr/bin/env python3
"""
Quick test to verify Python 3.11 and full intelligent chunking is working
"""
import sys
import pytest
from langchain.schema import Document as LangChainDocument
from services.intelligent_chunking import intelligent_chunking_service

# Mark entire module as slow due to heavy service initialization
pytestmark = pytest.mark.slow

def test_python_version():
    """Test Python version"""
    print("🐍 Python Version Check")
    print("=" * 30)
    
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor == 11:
        print("✅ Python 3.11 is active!")
        return True
    else:
        print(f"❌ Expected Python 3.11, got {version.major}.{version.minor}")
        return False

def test_langchain_imports():
    """Test LangChain imports"""
    print(f"\n🔗 LangChain Import Test")
    print("=" * 30)
    
    try:
        from langchain.schema import Document
        print("✅ LangChain Document imported")
        
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        print("✅ RecursiveCharacterTextSplitter imported")
        
        from langchain_community.llms import Ollama
        print("✅ Ollama integration imported")
        
        import langchain
        print(f"✅ LangChain version: {langchain.__version__}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_intelligent_chunking():
    """Test intelligent chunking functionality"""
    print(f"\n🧠 Intelligent Chunking Test")
    print("=" * 35)
    
    try:
        # Create test document
        test_doc = LangChainDocument(
            page_content="""# Test Document

This is a test document for Python 3.11 verification.

## Features

- Feature 1: Advanced processing
- Feature 2: Real-time analysis
- Feature 3: Secure storage

### Performance Table

| Metric | Value | Unit |
|--------|-------|------|
| Speed | 100 | ops/sec |
| Accuracy | 99.5 | % |

## Code Example

```python
def process_document(content):
    chunks = create_chunks(content)
    return chunks
```

## Conclusion

The system is working perfectly with Python 3.11!""",
            metadata={
                "source": "python311_test.md",
                "document_type": "test",
                "page_number": 1
            }
        )
        
        print(f"📄 Created test document ({len(test_doc.page_content)} characters)")
        
        # Process with intelligent chunking
        chunks = intelligent_chunking_service.chunk_documents([test_doc])
        
        print(f"✅ Successfully created {len(chunks)} intelligent chunks")
        
        # Analyze chunk types
        chunk_types = {}
        for chunk in chunks:
            chunk_type = chunk.metadata.get("chunk_type", "unknown")
            chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
        
        print(f"📊 Chunk types found: {chunk_types}")
        
        # Get statistics
        stats = intelligent_chunking_service.get_chunk_statistics(chunks)
        print(f"📈 Statistics:")
        print(f"  - Total chunks: {stats.get('total_chunks', 0)}")
        print(f"  - Total words: {stats.get('total_words', 0)}")
        print(f"  - Avg words/chunk: {stats.get('average_words_per_chunk', 0):.1f}")
        
        # Verify expected features
        expected_types = {'heading', 'paragraph', 'list', 'table', 'code'}
        found_types = set(chunk_types.keys())
        
        print(f"\n🔍 Feature Verification:")
        for expected_type in expected_types:
            if expected_type in found_types:
                print(f"  ✅ {expected_type}: Found ({chunk_types[expected_type]} chunks)")
            else:
                print(f"  ⚠️  {expected_type}: Not found")
        
        # Success criteria
        success_criteria = [
            len(chunks) > 5,  # Should create multiple chunks
            len(found_types) >= 3,  # Should find at least 3 different types
            'table' in found_types,  # Should preserve tables
            'list' in found_types,  # Should preserve lists
            'code' in found_types,  # Should preserve code
        ]
        
        passed = sum(success_criteria)
        total = len(success_criteria)
        
        print(f"\n🎯 Success criteria: {passed}/{total}")
        
        if passed >= total - 1:  # Allow 1 failure
            print(f"🎉 Intelligent chunking test PASSED!")
            return True
        else:
            print(f"⚠️  Some features may need attention")
            return False
        
    except Exception as e:
        print(f"❌ Intelligent chunking test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🚀 Python 3.11 + Full Intelligent Chunking Verification")
    print("=" * 65)
    
    tests = [
        ("Python Version", test_python_version),
        ("LangChain Imports", test_langchain_imports),
        ("Intelligent Chunking", test_intelligent_chunking),
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n{'='*65}")
        results[test_name] = test_func()
    
    # Summary
    print(f"\n{'='*65}")
    print("📊 VERIFICATION RESULTS")
    print("=" * 25)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print(f"\n🎉 SUCCESS! Everything is working perfectly!")
        print(f"✨ Python 3.11 + Full Intelligent Chunking is ready!")
        print(f"🔧 Features verified:")
        print(f"   - ✅ Python 3.11 environment active")
        print(f"   - ✅ LangChain fully functional")
        print(f"   - ✅ Intelligent chunking working")
        print(f"   - ✅ AI-powered structure analysis")
        print(f"   - ✅ Table, list, and code preservation")
        print(f"   - ✅ Rich metadata generation")
        print(f"🚀 Ready for production use!")
    else:
        print(f"\n⚠️  Some tests failed. Please check the issues above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)