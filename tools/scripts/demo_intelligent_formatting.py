#!/usr/bin/env python3
"""
Demo script for intelligent response formatting
Shows the dual-model approach in action with practical examples
"""
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from services.intelligent_response_formatter import intelligent_response_formatter


def demo_table_formatting():
    """Demo table formatting with sales data"""
    print("📊 TABLE FORMATTING DEMO")
    print("=" * 50)
    
    query = "Create a table comparing our quarterly sales performance"
    raw_response = """
    Our quarterly sales performance shows strong growth:
    Q1 2024: Revenue $150,000, Units 1,200, Profit Margin 15%
    Q2 2024: Revenue $175,000, Units 1,400, Profit Margin 18%
    Q3 2024: Revenue $200,000, Units 1,600, Profit Margin 20%
    Q4 2024: Revenue $225,000, Units 1,800, Profit Margin 22%
    
    The trend shows consistent growth in all metrics throughout the year.
    """
    
    print(f"Query: '{query}'")
    print(f"Raw Response: {raw_response[:100]}...")
    
    try:
        formatted_response = intelligent_response_formatter.format_response_with_streamlit_components(
            query, raw_response
        )
        
        print(f"\n✅ Detected Format: {formatted_response.response_type.value}")
        print(f"✅ Has Table Data: {formatted_response.table_data is not None}")
        print(f"✅ Streamlit Components: {len(formatted_response.streamlit_components)}")
        
        print(f"\n📝 Formatted Response:\n{formatted_response.content}")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def demo_chart_formatting():
    """Demo chart formatting with trend data"""
    print("\n\n📈 CHART FORMATTING DEMO")
    print("=" * 50)
    
    query = "Show me a bar chart of our monthly website traffic"
    raw_response = """
    Website traffic data for the past 6 months:
    January: 45,000 visitors
    February: 52,000 visitors
    March: 48,000 visitors
    April: 61,000 visitors
    May: 58,000 visitors
    June: 67,000 visitors
    
    Traffic shows an overall upward trend with some seasonal variations.
    """
    
    print(f"Query: '{query}'")
    print(f"Raw Response: {raw_response[:100]}...")
    
    try:
        formatted_response = intelligent_response_formatter.format_response_with_streamlit_components(
            query, raw_response
        )
        
        print(f"\n✅ Detected Format: {formatted_response.response_type.value}")
        print(f"✅ Has Chart Data: {formatted_response.chart_data is not None}")
        print(f"✅ Chart Type: {formatted_response.chart_data.get('chart_type') if formatted_response.chart_data else 'None'}")
        
        print(f"\n📝 Formatted Response:\n{formatted_response.content}")
        
        if formatted_response.chart_data:
            print(f"\n📊 Chart Data:")
            print(f"Labels: {formatted_response.chart_data.get('labels', [])}")
            print(f"Values: {formatted_response.chart_data.get('values', [])}")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def demo_comparison_formatting():
    """Demo comparison formatting"""
    print("\n\n⚖️ COMPARISON FORMATTING DEMO")
    print("=" * 50)
    
    query = "Compare our three main products side by side"
    raw_response = """
    Product Alpha: Price $299, Features 15, Customer Rating 4.2/5, Market Share 35%
    Product Beta: Price $399, Features 22, Customer Rating 4.5/5, Market Share 28%
    Product Gamma: Price $199, Features 8, Customer Rating 3.9/5, Market Share 37%
    
    Alpha offers good value, Beta has premium features, Gamma is budget-friendly.
    """
    
    print(f"Query: '{query}'")
    print(f"Raw Response: {raw_response[:100]}...")
    
    try:
        formatted_response = intelligent_response_formatter.format_response_with_streamlit_components(
            query, raw_response
        )
        
        print(f"\n✅ Detected Format: {formatted_response.response_type.value}")
        print(f"✅ Comparison Requested: {formatted_response.metadata.get('comparison_requested', False)}")
        
        print(f"\n📝 Formatted Response:\n{formatted_response.content}")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def demo_process_flow_formatting():
    """Demo process flow formatting"""
    print("\n\n🔄 PROCESS FLOW FORMATTING DEMO")
    print("=" * 50)
    
    query = "How do I set up the new customer onboarding process?"
    raw_response = """
    Setting up customer onboarding involves several steps:
    First, create the customer profile in our CRM system
    Then, send the welcome email with login credentials
    Next, schedule the initial consultation call
    After that, provide access to the training materials
    Finally, assign a dedicated account manager
    
    The entire process should take 3-5 business days to complete.
    """
    
    print(f"Query: '{query}'")
    print(f"Raw Response: {raw_response[:100]}...")
    
    try:
        formatted_response = intelligent_response_formatter.format_response_with_streamlit_components(
            query, raw_response
        )
        
        print(f"\n✅ Detected Format: {formatted_response.response_type.value}")
        print(f"✅ Process Flow Requested: {formatted_response.metadata.get('process_flow_requested', False)}")
        
        print(f"\n📝 Formatted Response:\n{formatted_response.content}")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def main():
    """Run all demos"""
    print("🎯 INTELLIGENT RESPONSE FORMATTING DEMO")
    print("=" * 60)
    print("This demo shows how the dual-model approach works:")
    print("• llava:7b analyzes query intent")
    print("• qwen3:4b generates formatted responses")
    print("• Streamlit components render the results")
    print("=" * 60)
    
    demos = [
        demo_table_formatting,
        demo_chart_formatting,
        demo_comparison_formatting,
        demo_process_flow_formatting
    ]
    
    for demo in demos:
        try:
            demo()
        except Exception as e:
            print(f"❌ Demo failed: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 Demo completed! The intelligent formatting system can:")
    print("✅ Detect query intent automatically")
    print("✅ Format responses appropriately")
    print("✅ Generate Streamlit components")
    print("✅ Extract data for tables and charts")
    print("✅ Provide fallback formatting")
    print("=" * 60)


if __name__ == "__main__":
    main()