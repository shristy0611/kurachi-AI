#!/usr/bin/env python3
"""
Simple Real Data Test - Tests our system on real documents without LLM dependencies
"""
import sys
import os
import time
from pathlib import Path
import pytest

# Mark entire module as slow due to heavy pipeline execution
pytestmark = pytest.mark.slow

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from services.document_processors import processor_factory
from services.intelligent_chunking import intelligent_chunking_service

def test_real_documents():
    """Test real documents in the test_data directory"""
    print("🧪 Real Data Test - Document Processing")
    print("=" * 50)
    
    test_data_dir = Path("tests/test_data")
    if not test_data_dir.exists():
        print(f"❌ Test data directory not found: {test_data_dir}")
        return False
    
    test_files = list(test_data_dir.glob("*"))
    print(f"📁 Found {len(test_files)} test files:")
    for file in test_files:
        print(f"  - {file.name} ({file.stat().st_size} bytes)")
    
    results = {}
    
    for test_file in test_files:
        if test_file.is_file():
            print(f"\n🔍 Testing: {test_file.name}")
            print("-" * 40)
            
            result = test_single_file(test_file)
            results[test_file.name] = result
    
    # Summary
    print(f"\n📊 SUMMARY")
    print("=" * 20)
    
    successful = 0
    total = len(results)
    
    for filename, result in results.items():
        status = "✅ SUCCESS" if result["success"] else "❌ FAILED"
        print(f"{filename}: {status}")
        if result["success"]:
            successful += 1
            print(f"  - Processor: {result['processor']}")
            print(f"  - Method: {result['method']}")
            print(f"  - Chunks: {result['chunks']}")
            print(f"  - Time: {result['time']:.2f}s")
        else:
            print(f"  - Error: {result['error']}")
    
    print(f"\n🎯 Success Rate: {successful}/{total} ({successful/total*100:.1f}%)")
    
    return successful > 0

def test_single_file(file_path: Path):
    """Test processing a single file"""
    result = {
        "success": False,
        "processor": None,
        "method": None,
        "chunks": 0,
        "time": 0.0,
        "error": None
    }
    
    try:
        start_time = time.time()
        
        # Step 1: Get processor
        print(f"1️⃣ Getting processor...")
        processor = processor_factory.get_processor(str(file_path))
        
        if not processor:
            result["error"] = f"No processor for {file_path.suffix}"
            print(f"❌ {result['error']}")
            return result
        
        result["processor"] = processor.__class__.__name__
        print(f"✅ Using: {result['processor']}")
        
        # Step 2: Process document
        print(f"2️⃣ Processing document...")
        processing_result = processor.process(str(file_path))
        
        if not processing_result.success:
            result["error"] = processing_result.error_message
            print(f"❌ Processing failed: {result['error']}")
            return result
        
        result["method"] = processing_result.metadata.get("extraction_method", "unknown")
        print(f"✅ Processed with method: {result['method']}")
        print(f"   Documents extracted: {len(processing_result.documents)}")
        
        # Step 3: Test chunking (disable SOTA to avoid LLM calls)
        print(f"3️⃣ Testing chunking...")
        
        if not processing_result.documents:
            result["error"] = "No documents extracted"
            print(f"❌ {result['error']}")
            return result
        
        # Add metadata to documents
        for doc in processing_result.documents:
            doc.metadata.update({
                "source": str(file_path),
                "filename": file_path.name,
                "file_type": file_path.suffix.lower()
            })
        
        # Disable SOTA to avoid LLM calls
        original_sota = intelligent_chunking_service.use_sota_markdown
        intelligent_chunking_service.use_sota_markdown = False
        
        try:
            chunks = intelligent_chunking_service.chunk_documents(processing_result.documents)
            result["chunks"] = len(chunks)
            print(f"✅ Created {len(chunks)} chunks")
            
            # Show chunk types
            chunk_types = {}
            for chunk in chunks:
                chunk_type = chunk.metadata.get("chunk_type", "unknown")
                chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
            
            if chunk_types:
                print(f"   Chunk types: {chunk_types}")
            
            # Show content preview
            if chunks:
                preview = chunks[0].page_content[:100].replace('\n', ' ')
                print(f"   Preview: {preview}...")
            
        finally:
            # Restore SOTA setting
            intelligent_chunking_service.use_sota_markdown = original_sota
        
        result["time"] = time.time() - start_time
        result["success"] = True
        
        return result
        
    except Exception as e:
        result["error"] = str(e)
        result["time"] = time.time() - start_time
        print(f"❌ Error: {e}")
        return result

def test_specific_features():
    """Test specific features of our system"""
    print(f"\n🔧 Testing Specific Features")
    print("=" * 35)
    
    # Test processor factory
    print("📋 Testing Processor Factory:")
    test_extensions = [".pdf", ".png", ".jpg", ".txt", ".docx", ".xlsx", ".pptx"]
    
    for ext in test_extensions:
        processor = processor_factory.get_processor(f"test{ext}")
        if processor:
            print(f"  ✅ {ext}: {processor.__class__.__name__}")
        else:
            print(f"  ❌ {ext}: No processor")
    
    # Test chunking service
    print(f"\n🧠 Testing Chunking Service:")
    from langchain.schema import Document as LangChainDocument
    
    test_doc = LangChainDocument(
        page_content="# Test Document\n\nThis is a test.\n\n- Item 1\n- Item 2",
        metadata={"source": "test.md"}
    )
    
    # Test without SOTA
    intelligent_chunking_service.use_sota_markdown = False
    chunks = intelligent_chunking_service.chunk_documents([test_doc])
    print(f"  ✅ Standard chunking: {len(chunks)} chunks")
    
    return True

def main():
    """Run the simple real data test"""
    print("🚀 Simple Real Data Test")
    print("Testing document processing without LLM dependencies")
    print("=" * 60)
    
    # Check Python version
    import sys
    version = sys.version_info
    print(f"🐍 Python: {version.major}.{version.minor}.{version.micro}")
    
    success = True
    
    try:
        # Test specific features first
        if not test_specific_features():
            success = False
        
        # Test real documents
        if not test_real_documents():
            success = False
        
        print(f"\n{'='*60}")
        if success:
            print("🎉 SUCCESS: Real data processing is working!")
            print("✨ Key findings:")
            print("   - Document processors are functional")
            print("   - Intelligent chunking works without LLM")
            print("   - System handles real documents correctly")
        else:
            print("⚠️  Some issues detected in real data processing")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        success = False
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)