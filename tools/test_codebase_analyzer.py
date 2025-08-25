#!/usr/bin/env python3
"""
Test script for the codebase analyzer to validate functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.codebase_analyzer import CodebaseAnalyzer, FileAnalysis
import json

def test_analyzer():
    """Test the codebase analyzer functionality"""
    print("Testing Codebase Analyzer...")
    
    # Initialize analyzer
    analyzer = CodebaseAnalyzer()
    
    # Test 1: Performance data loading
    print(f"✓ Performance data loaded: {len(analyzer.performance_data)} entries")
    assert len(analyzer.performance_data) > 0, "Should load performance data"
    
    # Test 2: Codebase analysis
    analyses = analyzer.analyze_codebase()
    print(f"✓ Analyzed {len(analyses)} files")
    assert len(analyses) > 0, "Should analyze files"
    
    # Test 3: File categorization
    categories = set(analysis.category for analysis in analyses)
    expected_categories = {'test_runner', 'analysis', 'test', 'source'}
    assert expected_categories.issubset(categories), f"Missing categories: {expected_categories - categories}"
    print(f"✓ File categorization working: {categories}")
    
    # Test 4: Test runner identification and scoring
    test_runners = [a for a in analyses if a.category == 'test_runner']
    print(f"✓ Found {len(test_runners)} test runners")
    assert len(test_runners) >= 5, "Should find multiple test runners"
    
    # Verify ultra_fast has highest score
    ultra_fast_runner = next((a for a in test_runners if 'ultra_fast' in a.path), None)
    if ultra_fast_runner:
        print(f"✓ Ultra fast runner score: {ultra_fast_runner.performance_score}")
        assert ultra_fast_runner.performance_score >= 9.0, "Ultra fast should have high score"
    
    # Test 5: Duplicate detection
    similar_files_found = any(len(a.similar_files) > 0 for a in analyses)
    print(f"✓ Similar files detection: {'Working' if similar_files_found else 'No similar files found'}")
    
    # Test 6: Recommendations
    recommendations = set(analysis.recommendation for analysis in analyses)
    expected_recs = {'keep', 'remove', 'rename'}
    found_recs = recommendations & expected_recs
    print(f"✓ Recommendations generated: {found_recs}")
    assert len(found_recs) > 0, "Should generate recommendations"
    
    # Test 7: Optimization plan
    plan = analyzer.get_optimization_plan()
    print(f"✓ Optimization plan: {len(plan.files_to_remove)} to remove, {len(plan.files_to_rename)} to rename")
    
    # Test 8: Analysis report generation
    report = analyzer.save_analysis_report("test_analysis_report.json")
    print(f"✓ Analysis report saved with {report['analysis_summary']['total_files']} files")
    
    # Verify report structure
    assert 'analysis_summary' in report
    assert 'file_analyses' in report
    assert 'optimization_plan' in report
    
    print("\n=== Test Results Summary ===")
    print(f"Total files analyzed: {len(analyses)}")
    print(f"Test runners found: {len(test_runners)}")
    print(f"Analysis files found: {len([a for a in analyses if a.category == 'analysis'])}")
    print(f"Files recommended for removal: {len(plan.files_to_remove)}")
    print(f"Files recommended for renaming: {len(plan.files_to_rename)}")
    
    # Show specific test runner scores
    print(f"\nTest Runner Performance Scores:")
    for runner in sorted(test_runners, key=lambda x: x.performance_score, reverse=True):
        print(f"  {runner.path}: {runner.performance_score}")
    
    print("\n✅ All tests passed! Codebase analyzer is working correctly.")
    
    # Clean up test file
    if os.path.exists("test_analysis_report.json"):
        os.remove("test_analysis_report.json")
    
    return True

if __name__ == "__main__":
    test_analyzer()