#!/usr/bin/env python3
"""
Test script for enhanced source attribution functionality
Tests the improvements made to task 4.2: Enhance response generation with source attribution
"""
import sys
from pathlib import Path
import pytest

# Mark as slow due to heavy service initialization
pytestmark = pytest.mark.slow

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from services.source_attribution import source_attribution_service, SourceCitation, ResponseAttribution
from langchain.schema import Document
from utils.logger import get_logger

logger = get_logger("test_source_attribution")


def test_enhanced_confidence_scoring():
    """Test enhanced confidence scoring with multiple factors"""
    print("🧪 Testing Enhanced Confidence Scoring...")
    
    # Create test documents
    test_docs = [
        Document(
            page_content="The quarterly revenue for Q3 2023 was $2.5 million, representing a 15% increase from the previous quarter. This growth was driven by strong performance in the software division.",
            metadata={"filename": "Q3_Report.pdf", "page": 5, "document_id": "doc1"}
        ),
        Document(
            page_content="Software division performance has been exceptional this year, with revenue growth of 15% in Q3. The team has successfully launched three new products.",
            metadata={"filename": "Software_Analysis.pdf", "page": 12, "document_id": "doc2"}
        )
    ]
    
    test_response = "The Q3 2023 revenue was $2.5 million, showing 15% growth primarily due to software division performance and new product launches."
    test_query = "What was the Q3 revenue and what drove the growth?"
    
    # Test enhanced attribution
    attribution = source_attribution_service.enhance_response_with_attribution(
        test_response, test_docs, test_query
    )
    
    print(f"✅ Overall Confidence: {attribution.overall_confidence:.2%}")
    print(f"✅ Fact Check Status: {attribution.fact_check_status}")
    print(f"✅ Synthesis Type: {attribution.synthesis_type}")
    print(f"✅ Sources Found: {len(attribution.sources)}")
    print(f"✅ Uncertainty Indicators: {len(attribution.uncertainty_indicators)}")
    
    # Test individual citation confidence scores
    for i, citation in enumerate(attribution.sources):
        print(f"   Source {i+1}: {citation.filename} - Confidence: {citation.confidence_score:.2%}, Relevance: {citation.relevance_score:.2%}")
    
    return attribution.overall_confidence > 0.7  # Should be high confidence


def test_multi_document_synthesis():
    """Test multi-document synthesis analysis"""
    print("\n🧪 Testing Multi-Document Synthesis...")
    
    # Create diverse test documents
    test_docs = [
        Document(
            page_content="Market analysis shows strong demand for AI solutions in healthcare. The sector is expected to grow by 25% annually.",
            metadata={"filename": "Market_Research.pdf", "page": 3, "document_id": "doc1"}
        ),
        Document(
            page_content="Our healthcare AI product has gained significant traction with 50 new clients this quarter. Revenue from healthcare segment increased 30%.",
            metadata={"filename": "Sales_Report.pdf", "page": 8, "document_id": "doc2"}
        ),
        Document(
            page_content="Healthcare AI implementation challenges include data privacy concerns and regulatory compliance. However, ROI is typically 200-300%.",
            metadata={"filename": "Implementation_Guide.pdf", "page": 15, "document_id": "doc3"}
        )
    ]
    
    test_response = "The healthcare AI market is experiencing strong growth at 25% annually, which aligns with our 30% revenue increase and 50 new clients. While implementation faces privacy and compliance challenges, ROI remains strong at 200-300%."
    test_query = "What's the status of healthcare AI market and our performance?"
    
    attribution = source_attribution_service.enhance_response_with_attribution(
        test_response, test_docs, test_query
    )
    
    print(f"✅ Synthesis Quality: {attribution.synthesis_type}")
    print(f"✅ Source Diversity: Multiple documents from different perspectives")
    print(f"✅ Information Integration: {len(attribution.sources)} sources integrated")
    print(f"✅ Fact Check Status: {attribution.fact_check_status}")
    
    return len(attribution.sources) >= 3 and "multi" in attribution.synthesis_type.lower()


def test_uncertainty_indicators():
    """Test enhanced uncertainty indicator detection"""
    print("\n🧪 Testing Enhanced Uncertainty Indicators...")
    
    # Create test with uncertain response
    test_docs = [
        Document(
            page_content="The project timeline is approximately 6 months, but this may vary depending on resource availability.",
            metadata={"filename": "Project_Plan.pdf", "page": 2, "document_id": "doc1"}
        )
    ]
    
    test_response = "The project timeline appears to be around 6 months, though this might change based on available resources. It's unclear if additional factors could affect the schedule."
    test_query = "How long will the project take?"
    
    attribution = source_attribution_service.enhance_response_with_attribution(
        test_response, test_docs, test_query
    )
    
    print(f"✅ Uncertainty Indicators Found: {len(attribution.uncertainty_indicators)}")
    for indicator in attribution.uncertainty_indicators:
        print(f"   - {indicator}")
    
    print(f"✅ Confidence Impact: {attribution.overall_confidence:.2%} (should be lower due to uncertainty)")
    
    return len(attribution.uncertainty_indicators) > 0 and attribution.overall_confidence < 0.7


def test_fact_checking_validation():
    """Test comprehensive fact-checking"""
    print("\n🧪 Testing Comprehensive Fact-Checking...")
    
    # Create test with verifiable facts
    test_docs = [
        Document(
            page_content="The company was founded in 2020 with initial funding of $5 million. Current employee count is 150 people.",
            metadata={"filename": "Company_Info.pdf", "page": 1, "document_id": "doc1"}
        ),
        Document(
            page_content="As of 2023, the company has grown to 150 employees and has raised a total of $25 million in funding.",
            metadata={"filename": "Annual_Report.pdf", "page": 4, "document_id": "doc2"}
        )
    ]
    
    test_response = "The company was founded in 2020 with $5 million initial funding and now has 150 employees with total funding of $25 million."
    test_query = "Tell me about the company's founding and current status"
    
    attribution = source_attribution_service.enhance_response_with_attribution(
        test_response, test_docs, test_query
    )
    
    print(f"✅ Fact Check Status: {attribution.fact_check_status}")
    print(f"✅ Validation Notes: {attribution.validation_notes}")
    print(f"✅ Confidence: {attribution.overall_confidence:.2%}")
    
    return attribution.fact_check_status in ["verified", "mostly_verified"]


def test_citation_formatting():
    """Test citation formatting for display"""
    print("\n🧪 Testing Citation Formatting...")
    
    # Create test citation
    citation = SourceCitation(
        document_id="doc1",
        filename="Business_Plan.pdf",
        page_number=15,
        section="Financial Projections",
        excerpt="Revenue is projected to reach $10 million by year 3, with a gross margin of 65%.",
        relevance_score=0.85,
        confidence_score=0.92,
        citation_type="direct",
        metadata={"detected_language": "english"}
    )
    
    formatted = source_attribution_service.format_citations_for_display([citation])
    print("✅ Formatted Citation:")
    print(formatted)
    
    # Test confidence explanation
    attribution = ResponseAttribution(
        response_content="Test response",
        overall_confidence=0.85,
        sources=[citation],
        fact_check_status="verified",
        synthesis_type="single_source"
    )
    
    explanation = source_attribution_service.generate_confidence_explanation(attribution)
    print("\n✅ Confidence Explanation:")
    print(explanation)
    
    return "Business_Plan.pdf" in formatted and "Page 15" in formatted


def run_all_tests():
    """Run all enhanced source attribution tests"""
    print("🚀 Starting Enhanced Source Attribution Tests")
    print("=" * 60)
    
    tests = [
        ("Enhanced Confidence Scoring", test_enhanced_confidence_scoring),
        ("Multi-Document Synthesis", test_multi_document_synthesis),
        ("Uncertainty Indicators", test_uncertainty_indicators),
        ("Fact-Checking Validation", test_fact_checking_validation),
        ("Citation Formatting", test_citation_formatting)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result, None))
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"\n{status}: {test_name}")
        except Exception as e:
            results.append((test_name, False, str(e)))
            print(f"\n❌ ERROR: {test_name} - {str(e)}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result, _ in results if result)
    total = len(results)
    
    for test_name, result, error in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {test_name}")
        if error:
            print(f"   Error: {error}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All tests passed! Enhanced source attribution is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please review the implementation.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)