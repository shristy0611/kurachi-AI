#!/usr/bin/env python3
"""
Test script for intelligent response formatting
Tests the dual-model approach with llava:7b and qwen3:4b
"""
import sys
import os
from pathlib import Path
import pytest

# Mark as slow due to model initialization and formatting services
pytestmark = pytest.mark.slow

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from services.intelligent_response_formatter import intelligent_response_formatter, ResponseType, ChartType
from utils.logger import get_logger

logger = get_logger("test_intelligent_formatting")


def test_query_intent_analysis():
    """Test query intent analysis with various query types"""
    print("🧠 Testing Query Intent Analysis")
    print("=" * 50)
    
    test_queries = [
        # Table requests
        ("Show me a comparison table of sales data", ResponseType.TABLE),
        ("Can you create a table with the quarterly results?", ResponseType.TABLE),
        ("List the data in tabular format", ResponseType.TABLE),
        
        # Chart requests
        ("Create a bar chart of the revenue trends", ResponseType.CHART),
        ("Show me a graph of the performance metrics", ResponseType.CHART),
        ("Visualize the data as a pie chart", ResponseType.CHART),
        
        # List requests
        ("List all the features", ResponseType.LIST),
        ("What are the main benefits?", ResponseType.LIST),
        ("Show me the available options", ResponseType.LIST),
        
        # Comparison requests
        ("Compare product A vs product B", ResponseType.COMPARISON),
        ("What's the difference between these two approaches?", ResponseType.COMPARISON),
        ("Which is better: option 1 or option 2?", ResponseType.COMPARISON),
        
        # Process flow requests
        ("How do I set up the system?", ResponseType.PROCESS_FLOW),
        ("What are the steps to complete this task?", ResponseType.PROCESS_FLOW),
        ("Walk me through the installation process", ResponseType.PROCESS_FLOW),
        
        # Summary requests
        ("Give me a brief overview", ResponseType.SUMMARY),
        ("Summarize the key points", ResponseType.SUMMARY),
        ("What's the executive summary?", ResponseType.SUMMARY),
        
        # Code requests
        ("Show me a Python example", ResponseType.CODE_BLOCK),
        ("What's the syntax for this function?", ResponseType.CODE_BLOCK),
        ("Give me a code snippet", ResponseType.CODE_BLOCK),
    ]
    
    correct_predictions = 0
    total_predictions = len(test_queries)
    
    for query, expected_type in test_queries:
        try:
            intent = intelligent_response_formatter.analyze_query_intent(query)
            
            print(f"\nQuery: '{query}'")
            print(f"Expected: {expected_type.value}")
            print(f"Predicted: {intent.response_type.value}")
            print(f"Confidence: {intent.confidence:.2f}")
            print(f"Reasoning: {intent.reasoning}")
            
            if intent.response_type == expected_type:
                print("✅ CORRECT")
                correct_predictions += 1
            else:
                print("❌ INCORRECT")
                
        except Exception as e:
            print(f"❌ ERROR: {e}")
    
    accuracy = (correct_predictions / total_predictions) * 100
    print(f"\n📊 Overall Accuracy: {accuracy:.1f}% ({correct_predictions}/{total_predictions})")
    
    return accuracy > 70  # Expect at least 70% accuracy


def test_response_formatting():
    """Test response formatting with different content types"""
    print("\n🎨 Testing Response Formatting")
    print("=" * 50)
    
    test_cases = [
        {
            "query": "Show me a table of quarterly sales data",
            "raw_response": """
            Q1 2024: $150,000
            Q2 2024: $175,000
            Q3 2024: $200,000
            Q4 2024: $225,000
            
            The sales have been steadily increasing throughout the year.
            """,
            "expected_type": ResponseType.TABLE
        },
        {
            "query": "Create a chart showing revenue trends",
            "raw_response": """
            January: $50,000
            February: $55,000
            March: $60,000
            April: $58,000
            May: $65,000
            
            Revenue has shown an overall upward trend with slight fluctuations.
            """,
            "expected_type": ResponseType.CHART
        },
        {
            "query": "List the main features of the product",
            "raw_response": """
            The product includes several key features:
            Advanced analytics capabilities
            Real-time data processing
            User-friendly interface
            Scalable architecture
            24/7 customer support
            """,
            "expected_type": ResponseType.LIST
        },
        {
            "query": "How do I install the software?",
            "raw_response": """
            To install the software, follow these steps:
            Download the installer from our website
            Run the installer as administrator
            Follow the setup wizard
            Enter your license key
            Complete the installation
            Restart your computer
            """,
            "expected_type": ResponseType.PROCESS_FLOW
        }
    ]
    
    successful_formats = 0
    
    for i, test_case in enumerate(test_cases, 1):
        try:
            print(f"\nTest Case {i}:")
            print(f"Query: '{test_case['query']}'")
            
            # Analyze intent
            intent = intelligent_response_formatter.analyze_query_intent(test_case["query"])
            print(f"Detected Type: {intent.response_type.value}")
            print(f"Expected Type: {test_case['expected_type'].value}")
            
            # Generate formatted response
            formatted_response = intelligent_response_formatter.generate_formatted_response(
                test_case["query"],
                test_case["raw_response"],
                intent
            )
            
            print(f"Response Type: {formatted_response.response_type.value}")
            print(f"Has Table Data: {formatted_response.table_data is not None}")
            print(f"Has Chart Data: {formatted_response.chart_data is not None}")
            print(f"Streamlit Components: {len(formatted_response.streamlit_components)}")
            
            # Show formatted content (first 200 chars)
            content_preview = formatted_response.content[:200] + "..." if len(formatted_response.content) > 200 else formatted_response.content
            print(f"Formatted Content Preview:\n{content_preview}")
            
            if formatted_response.response_type != ResponseType.PLAIN_TEXT:
                print("✅ FORMATTING APPLIED")
                successful_formats += 1
            else:
                print("⚠️ NO SPECIAL FORMATTING")
                
        except Exception as e:
            print(f"❌ ERROR: {e}")
    
    success_rate = (successful_formats / len(test_cases)) * 100
    print(f"\n📊 Formatting Success Rate: {success_rate:.1f}% ({successful_formats}/{len(test_cases)})")
    
    return success_rate > 50  # Expect at least 50% success rate


def test_end_to_end_formatting():
    """Test end-to-end formatting with complete workflow"""
    print("\n🔄 Testing End-to-End Formatting")
    print("=" * 50)
    
    test_query = "Compare the performance of our three main products in a table format"
    raw_response = """
    Product A has achieved 85% customer satisfaction with 1,200 units sold and $240,000 revenue.
    Product B shows 92% customer satisfaction with 800 units sold and $320,000 revenue.
    Product C demonstrates 78% customer satisfaction with 1,500 units sold and $180,000 revenue.
    
    Product B has the highest customer satisfaction and revenue per unit, while Product C has the highest total units sold.
    """
    
    try:
        # Complete formatting workflow
        formatted_response = intelligent_response_formatter.format_response_with_streamlit_components(
            test_query, raw_response
        )
        
        print(f"Query: '{test_query}'")
        print(f"Detected Response Type: {formatted_response.response_type.value}")
        print(f"Has Streamlit Components: {len(formatted_response.streamlit_components) > 0}")
        
        # Show metadata
        metadata = formatted_response.metadata
        print(f"Intent Confidence: {metadata.get('intent_confidence', 0):.2f}")
        print(f"Intent Reasoning: {metadata.get('intent_reasoning', 'N/A')}")
        
        # Show formatted content
        print(f"\nFormatted Content:\n{formatted_response.content}")
        
        # Show component info
        if formatted_response.streamlit_components:
            print(f"\nStreamlit Components:")
            for component in formatted_response.streamlit_components:
                print(f"- Type: {component.get('type')}")
        
        print("✅ END-TO-END TEST COMPLETED")
        return True
        
    except Exception as e:
        print(f"❌ END-TO-END TEST FAILED: {e}")
        return False


def main():
    """Run all tests"""
    print("🧪 Testing Intelligent Response Formatter")
    print("=" * 60)
    
    # Check if Ollama is running
    try:
        # Test basic functionality
        intent = intelligent_response_formatter.analyze_query_intent("test query")
        print("✅ Ollama connection successful")
    except Exception as e:
        print(f"❌ Ollama connection failed: {e}")
        print("Please ensure Ollama is running with llava:7b and qwen3:4b models")
        return False
    
    # Run tests
    tests = [
        ("Query Intent Analysis", test_query_intent_analysis),
        ("Response Formatting", test_response_formatting),
        ("End-to-End Formatting", test_end_to_end_formatting)
    ]
    
    passed_tests = 0
    
    for test_name, test_func in tests:
        try:
            print(f"\n{'='*20} {test_name} {'='*20}")
            if test_func():
                print(f"✅ {test_name} PASSED")
                passed_tests += 1
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")
    
    # Final results
    print(f"\n{'='*60}")
    print(f"📊 Test Results: {passed_tests}/{len(tests)} tests passed")
    
    if passed_tests == len(tests):
        print("🎉 All tests passed! Intelligent response formatting is working correctly.")
        return True
    else:
        print("⚠️ Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)