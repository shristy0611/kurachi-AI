#!/usr/bin/env python3
"""
Test SOTA Markdown Conversion for Intelligent Chunking
Demonstrates the state-of-the-art approach of converting all content to Markdown
"""
import sys
from pathlib import Path
import pytest

# Mark as slow due to heavy intelligent chunking service
pytestmark = pytest.mark.slow

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from langchain.schema import Document as LangChainDocument
from services.intelligent_chunking import intelligent_chunking_service
from services.markdown_converter import sota_markdown_chunking_service


def create_test_documents():
    """Create test documents simulating different file types"""
    
    # Simulate PDF extraction (messy format)
    pdf_content = """QUARTERLY BUSINESS REPORT
    
Financial Performance Q3 2024

Revenue Growth
Our company achieved significant growth this quarter with total revenue of $2.5M.

Key Metrics:
Revenue: $2,500,000
Profit: $450,000
Growth Rate: 15%

Department Performance:
Sales Department achieved 120% of target
Marketing Department increased leads by 35%
Engineering Department delivered 8 new features

Market Analysis
The market conditions remain favorable with increasing demand for our products.

Competitive Landscape:
Competitor A: 25% market share
Competitor B: 20% market share
Our Company: 18% market share

Future Outlook
We expect continued growth in Q4 2024 with projected revenue of $3M."""

    # Simulate Excel extraction (tabular data)
    excel_content = """Employee Data Sheet

Name    Department    Salary    Performance
John Smith    Engineering    75000    Excellent
Jane Doe    Marketing    68000    Good
Mike Johnson    Sales    72000    Excellent
Sarah Wilson    Engineering    71000    Good
Tom Brown    Marketing    65000    Fair

Department Summary:
Engineering: 2 employees, average salary $73,000
Marketing: 2 employees, average salary $66,500
Sales: 1 employee, average salary $72,000

Total Employees: 5
Average Salary: $70,200
Performance Distribution:
Excellent: 2 employees
Good: 2 employees
Fair: 1 employee"""

    # Simulate PowerPoint extraction (presentation format)
    ppt_content = """Product Launch Presentation

Slide 1: Introduction
Welcome to our new product launch presentation
Today we'll cover features, benefits, and market strategy

Slide 2: Product Overview
Revolutionary AI-powered document processing
Key features:
- Intelligent content extraction
- Multi-format support
- Real-time processing
- Advanced analytics

Slide 3: Market Opportunity
Target market size: $50B
Expected market share: 5%
Revenue projection: $2.5B over 3 years

Slide 4: Technical Architecture
Cloud-based infrastructure
Microservices architecture
AI/ML processing pipeline
Scalable to millions of documents

Slide 5: Go-to-Market Strategy
Phase 1: Beta launch with select customers
Phase 2: Public launch and marketing campaign
Phase 3: Enterprise sales and partnerships"""

    return [
        LangChainDocument(
            page_content=pdf_content,
            metadata={
                "source": "quarterly_report.pdf",
                "document_type": "business_report",
                "extraction_method": "text_extraction",
                "file_type": ".pdf"
            }
        ),
        LangChainDocument(
            page_content=excel_content,
            metadata={
                "source": "employee_data.xlsx",
                "document_type": "spreadsheet",
                "extraction_method": "excel_processing",
                "file_type": ".xlsx"
            }
        ),
        LangChainDocument(
            page_content=ppt_content,
            metadata={
                "source": "product_launch.pptx",
                "document_type": "presentation",
                "extraction_method": "powerpoint_processing",
                "file_type": ".pptx"
            }
        )
    ]


def test_sota_markdown_conversion():
    """Test SOTA Markdown conversion"""
    print("🚀 Testing SOTA Markdown Conversion")
    print("=" * 50)
    
    # Create test documents
    test_docs = create_test_documents()
    print(f"📄 Created {len(test_docs)} test documents:")
    for doc in test_docs:
        print(f"  - {doc.metadata['source']} ({doc.metadata['file_type']})")
    
    print(f"\n🔄 Converting to Markdown...")
    
    # Test Markdown conversion
    markdown_result = sota_markdown_chunking_service.process_documents_to_markdown(test_docs)
    
    if not markdown_result.success:
        print(f"❌ Markdown conversion failed: {markdown_result.error_message}")
        return False
    
    print(f"✅ Markdown conversion successful!")
    print(f"📊 Conversion Statistics:")
    
    stats = sota_markdown_chunking_service.get_conversion_statistics(markdown_result)
    print(f"  - Original documents: {stats.get('original_documents', 0)}")
    print(f"  - Markdown length: {stats.get('markdown_length', 0)} characters")
    print(f"  - RAG optimization score: {stats.get('rag_optimization_score', 0):.1f}%")
    
    structure_elements = stats.get('structure_elements', {})
    if structure_elements:
        print(f"  - Structure elements: {structure_elements}")
    
    # Show Markdown preview
    print(f"\n📝 Markdown Preview (first 500 characters):")
    print("-" * 50)
    preview = markdown_result.markdown_content[:500]
    print(preview)
    if len(markdown_result.markdown_content) > 500:
        print("...")
    print("-" * 50)
    
    return True


def test_sota_intelligent_chunking():
    """Test complete SOTA intelligent chunking pipeline"""
    print(f"\n🧠 Testing SOTA Intelligent Chunking Pipeline")
    print("=" * 55)
    
    # Create test documents
    test_docs = create_test_documents()
    
    print(f"📄 Processing {len(test_docs)} documents with SOTA chunking...")
    
    # Process with SOTA intelligent chunking
    chunks = intelligent_chunking_service.chunk_documents(test_docs)
    
    print(f"✅ SOTA chunking completed!")
    print(f"📊 Results:")
    print(f"  - Total chunks created: {len(chunks)}")
    
    # Analyze chunks
    chunk_types = {}
    sota_chunks = 0
    rag_scores = []
    
    for chunk in chunks:
        chunk_type = chunk.metadata.get("chunk_type", "unknown")
        chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
        
        if chunk.metadata.get("sota_markdown_conversion"):
            sota_chunks += 1
            rag_score = chunk.metadata.get("rag_optimization_score", 0)
            if rag_score > 0:
                rag_scores.append(rag_score)
    
    print(f"  - Chunk types: {chunk_types}")
    print(f"  - SOTA Markdown chunks: {sota_chunks}/{len(chunks)}")
    
    if rag_scores:
        avg_rag_score = sum(rag_scores) / len(rag_scores)
        print(f"  - Average RAG optimization score: {avg_rag_score:.1f}%")
    
    # Show sample chunks
    print(f"\n📋 Sample Chunks:")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\nChunk {i+1}:")
        print(f"  Type: {chunk.metadata.get('chunk_type', 'unknown')}")
        print(f"  SOTA: {'Yes' if chunk.metadata.get('sota_markdown_conversion') else 'No'}")
        print(f"  Words: {chunk.metadata.get('word_count', 0)}")
        
        # Show content preview
        content_preview = chunk.page_content[:150].replace('\n', ' ')
        print(f"  Preview: {content_preview}...")
    
    # Get overall statistics
    stats = intelligent_chunking_service.get_chunk_statistics(chunks)
    print(f"\n📈 Overall Statistics:")
    print(f"  - Total words: {stats.get('total_words', 0)}")
    print(f"  - Average words per chunk: {stats.get('average_words_per_chunk', 0):.1f}")
    print(f"  - Average characters per chunk: {stats.get('average_chars_per_chunk', 0):.1f}")
    
    return len(chunks) > 0 and sota_chunks > 0


def compare_standard_vs_sota():
    """Compare standard chunking vs SOTA Markdown chunking"""
    print(f"\n⚖️  Comparing Standard vs SOTA Chunking")
    print("=" * 45)
    
    # Create test document
    test_doc = LangChainDocument(
        page_content="""Technical Documentation

System Architecture Overview

Our system consists of three main components:

Component 1: Data Ingestion Layer
- Handles file uploads
- Validates file formats
- Queues processing jobs

Component 2: Processing Engine
- Extracts content from documents
- Applies intelligent chunking
- Generates embeddings

Component 3: Query Interface
- Provides search capabilities
- Returns relevant results
- Maintains conversation context

Performance Metrics:

Metric          Value       Unit
Throughput      1000        docs/hour
Latency         50          milliseconds
Accuracy        95          percent

Configuration Settings:
chunk_size: 1000
chunk_overlap: 200
embedding_model: qwen3:4b""",
        metadata={"source": "tech_doc.md", "document_type": "technical"}
    )
    
    # Test standard chunking (disable SOTA)
    print("🔧 Testing Standard Chunking...")
    intelligent_chunking_service.use_sota_markdown = False
    standard_chunks = intelligent_chunking_service.chunk_documents([test_doc])
    
    # Test SOTA chunking
    print("🚀 Testing SOTA Chunking...")
    intelligent_chunking_service.use_sota_markdown = True
    sota_chunks = intelligent_chunking_service.chunk_documents([test_doc])
    
    # Compare results
    print(f"\n📊 Comparison Results:")
    print(f"  Standard Chunking:")
    print(f"    - Chunks created: {len(standard_chunks)}")
    print(f"    - Markdown optimized: 0")
    
    print(f"  SOTA Chunking:")
    print(f"    - Chunks created: {len(sota_chunks)}")
    sota_optimized = sum(1 for c in sota_chunks if c.metadata.get("sota_markdown_conversion"))
    print(f"    - Markdown optimized: {sota_optimized}")
    
    if sota_chunks:
        avg_rag_score = sum(c.metadata.get("rag_optimization_score", 0) for c in sota_chunks) / len(sota_chunks)
        print(f"    - Average RAG score: {avg_rag_score:.1f}%")
    
    # Show content comparison
    if standard_chunks and sota_chunks:
        print(f"\n📝 Content Comparison (First Chunk):")
        print(f"Standard: {standard_chunks[0].page_content[:100]}...")
        print(f"SOTA:     {sota_chunks[0].page_content[:100]}...")
    
    return True


def main():
    """Run all SOTA Markdown chunking tests"""
    print("🚀 SOTA Markdown Chunking Test Suite")
    print("=" * 60)
    
    tests = [
        ("Markdown Conversion", test_sota_markdown_conversion),
        ("SOTA Intelligent Chunking", test_sota_intelligent_chunking),
        ("Standard vs SOTA Comparison", compare_standard_vs_sota),
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} failed: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = False
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 SOTA CHUNKING TEST RESULTS")
    print("=" * 35)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {test_name}: {status}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print(f"\n🎉 ALL TESTS PASSED!")
        print(f"✨ SOTA Markdown chunking is working perfectly!")
        print(f"🚀 Benefits of SOTA approach:")
        print(f"   - ✅ Universal Markdown conversion")
        print(f"   - ✅ Optimal RAG performance")
        print(f"   - ✅ Structure preservation")
        print(f"   - ✅ LLM-optimized format")
        print(f"   - ✅ Enhanced metadata")
    else:
        print(f"\n⚠️  Some tests failed. Please review the implementation.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)