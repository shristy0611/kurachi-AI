#!/usr/bin/env python3
"""
Codebase Analysis and Categorization System

This module provides functionality to analyze the codebase, categorize files,
detect duplicates, and score performance based on test results data.
"""

import os
import json
import re
import hashlib
from dataclasses import dataclass, asdict
from typing import List, Dict, Set, Optional, Tuple
from pathlib import Path
import ast
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FileAnalysis:
    """Data model for file analysis results"""
    path: str
    category: str  # 'test_runner', 'analysis', 'test', 'source', 'config', 'docs'
    performance_score: float
    dependencies: List[str]
    similar_files: List[str]
    recommendation: str  # 'keep', 'remove', 'rename'
    file_size: int
    last_modified: float
    content_hash: str

@dataclass
class OptimizationPlan:
    """Data model for optimization plan"""
    files_to_remove: List[str]
    files_to_rename: Dict[str, str]  # old_name -> new_name
    total_files_analyzed: int
    potential_size_reduction: int
    performance_improvements: Dict[str, float]

class CodebaseAnalyzer:
    """Main analyzer class for codebase optimization"""
    
    def __init__(self, root_path: str = "."):
        self.root_path = Path(root_path)
        self.file_analyses: List[FileAnalysis] = []
        self.performance_data: Dict[str, float] = {}
        self.duplicate_groups: List[List[str]] = []
        
        # Load performance data from test analysis files
        self._load_performance_data()
        
        # Add known performance data for test runners based on analysis
        self._add_known_performance_data()
    
    def _load_performance_data(self):
        """Load performance data from existing test analysis files"""
        analysis_files = list(self.root_path.glob("*test_analysis*.json"))
        analysis_files.extend(list(self.root_path.glob("*_test_analysis*.json")))
        
        for analysis_file in analysis_files:
            try:
                with open(analysis_file, 'r') as f:
                    data = json.load(f)
                    
                # Extract performance metrics from different analysis file formats
                if 'test_results' in data:
                    for test_name, result in data['test_results'].items():
                        if isinstance(result, dict) and 'duration' in result:
                            self.performance_data[test_name] = result['duration']
                
                # Look for runner performance data
                if 'runner_performance' in data:
                    self.performance_data.update(data['runner_performance'])
                
                # Handle the current format with 'results' array
                if 'results' in data:
                    for result in data['results']:
                        if isinstance(result, dict) and 'name' in result and 'duration' in result:
                            self.performance_data[result['name']] = result['duration']
                
                # Handle summary data
                if 'summary' in data and isinstance(data['summary'], dict):
                    if 'total_time' in data['summary']:
                        # Use filename as key for overall performance
                        file_key = analysis_file.stem
                        self.performance_data[file_key] = data['summary']['total_time']
                    
                logger.info(f"Loaded performance data from {analysis_file}")
                
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Could not parse {analysis_file}: {e}")
    
    def _add_known_performance_data(self):
        """Add known performance data for test runners based on analysis"""
        # Based on the analysis files, we know these approximate performance metrics
        known_performance = {
            'test_runner_fast.py': 0.73,  # Best performer
            'run_fast_tests.py': 1.2,
            'run_optimized_complete_tests.py': 2.1,
            'run_complete_tests_with_analysis.py': 3.5,
            'run_tests.py': 4.0,  # Comprehensive but slower
            'run_minimal_tests.py': 1.8,
            'run_minimal_fast_tests.py': 1.1
        }
        
        # Only add if we don't already have performance data
        for runner, duration in known_performance.items():
            if runner not in self.performance_data:
                self.performance_data[runner] = duration
    
    def _get_file_hash(self, file_path: Path) -> str:
        """Calculate MD5 hash of file content for duplicate detection"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return ""
    
    def _extract_imports(self, file_path: Path) -> List[str]:
        """Extract import statements from Python files"""
        imports = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)
                        
        except Exception as e:
            logger.debug(f"Could not parse imports from {file_path}: {e}")
            
        return imports
    
    def _categorize_file(self, file_path: Path) -> str:
        """Categorize file based on path and content patterns"""
        path_str = str(file_path)
        name = file_path.name.lower()
        
        # Test runners
        if name.startswith('run_') and ('test' in name or 'runner' in name):
            return 'test_runner'
        
        # Analysis files
        if ('analysis' in name or 'summary' in name) and (name.endswith('.json') or name.endswith('.md')):
            return 'analysis'
        
        # Test files
        if name.startswith('test_') or '/tests/' in path_str:
            return 'test'
        
        # Configuration files
        if name.endswith(('.yml', '.yaml', '.json', '.ini', '.cfg', '.toml')):
            return 'config'
        
        # Documentation
        if name.endswith('.md') or '/docs/' in path_str:
            return 'docs'
        
        # Source code
        if name.endswith('.py'):
            return 'source'
        
        return 'other'
    
    def _calculate_performance_score(self, file_path: Path, category: str) -> float:
        """Calculate performance score based on category and available data"""
        name = file_path.name
        
        # For test runners, use actual performance data if available
        if category == 'test_runner':
            # Check for performance data by various name patterns
            for key in self.performance_data:
                if name in key or key in name:
                    # Lower duration = higher score (inverted)
                    duration = self.performance_data[key]
                    return max(0.1, 10.0 / max(duration, 0.1))
            
            # Default scoring based on name patterns for test runners
            if 'ultra_fast' in name:
                return 9.5
            elif 'fast' in name:
                return 8.0
            elif 'optimized' in name:
                return 7.0
            elif 'complete' in name:
                return 6.0
            elif 'minimal' in name:
                return 5.0
            else:
                return 4.0
        
        # For analysis files, prefer newer and more comprehensive ones
        elif category == 'analysis':
            if 'complete' in name.lower() or 'summary' in name.lower():
                return 8.0
            elif 'ultra_fast' in name.lower():
                return 7.0
            elif file_path.stat().st_mtime > 1724500000:  # Recent files
                return 6.0
            else:
                return 3.0
        
        # Default scoring for other categories
        return 5.0
    
    def _find_similar_files(self, file_path: Path, all_files: List[Path]) -> List[str]:
        """Find files with similar functionality or names"""
        similar = []
        name = file_path.name.lower()
        category = self._categorize_file(file_path)
        
        for other_file in all_files:
            if other_file == file_path:
                continue
                
            other_name = other_file.name.lower()
            other_category = self._categorize_file(other_file)
            
            # Same category files with similar names
            if category == other_category:
                # Test runners with overlapping functionality
                if category == 'test_runner':
                    common_words = set(name.split('_')) & set(other_name.split('_'))
                    if len(common_words) >= 2:  # At least 2 common words
                        similar.append(str(other_file))
                
                # Analysis files from similar time periods
                elif category == 'analysis':
                    if 'test_analysis' in name and 'test_analysis' in other_name:
                        similar.append(str(other_file))
        
        return similar
    
    def _generate_recommendation(self, analysis: FileAnalysis) -> str:
        """Generate recommendation based on analysis results"""
        # High-performing files should be kept
        if analysis.performance_score >= 8.0:
            return 'keep'
        
        # Files with many similar files and low performance should be removed
        if len(analysis.similar_files) >= 2 and analysis.performance_score < 6.0:
            return 'remove'
        
        # Files with non-standard naming should be renamed
        if analysis.category in ['test', 'test_runner']:
            if 'sota' in analysis.path.lower():
                return 'rename'
            if analysis.category == 'test_runner' and not analysis.path.startswith('test_runner_'):
                return 'rename'
        
        return 'keep'
    
    def analyze_codebase(self) -> List[FileAnalysis]:
        """Main method to analyze the entire codebase"""
        logger.info("Starting codebase analysis...")
        
        # Get all relevant files
        python_files = list(self.root_path.rglob("*.py"))
        analysis_files = list(self.root_path.rglob("*analysis*.json"))
        analysis_files.extend(list(self.root_path.rglob("*analysis*.md")))
        summary_files = list(self.root_path.rglob("*summary*.md"))
        
        all_files = python_files + analysis_files + summary_files
        
        # Filter out files in excluded directories
        excluded_dirs = {'.git', '__pycache__', '.pytest_cache', 'venv', 'venv311', 'node_modules'}
        all_files = [f for f in all_files if not any(part in excluded_dirs for part in f.parts)]
        
        logger.info(f"Analyzing {len(all_files)} files...")
        
        # Analyze each file
        for file_path in all_files:
            try:
                stat = file_path.stat()
                category = self._categorize_file(file_path)
                performance_score = self._calculate_performance_score(file_path, category)
                dependencies = self._extract_imports(file_path) if file_path.suffix == '.py' else []
                similar_files = self._find_similar_files(file_path, all_files)
                content_hash = self._get_file_hash(file_path)
                
                analysis = FileAnalysis(
                    path=str(file_path),
                    category=category,
                    performance_score=performance_score,
                    dependencies=dependencies,
                    similar_files=similar_files,
                    recommendation='keep',  # Will be updated below
                    file_size=stat.st_size,
                    last_modified=stat.st_mtime,
                    content_hash=content_hash
                )
                
                # Generate recommendation
                analysis.recommendation = self._generate_recommendation(analysis)
                
                self.file_analyses.append(analysis)
                
            except Exception as e:
                logger.warning(f"Could not analyze {file_path}: {e}")
        
        # Detect duplicate groups
        self._detect_duplicates()
        
        logger.info(f"Analysis complete. Analyzed {len(self.file_analyses)} files.")
        return self.file_analyses
    
    def _detect_duplicates(self):
        """Detect groups of duplicate or very similar files"""
        hash_groups = {}
        
        # Group by content hash
        for analysis in self.file_analyses:
            if analysis.content_hash:
                if analysis.content_hash not in hash_groups:
                    hash_groups[analysis.content_hash] = []
                hash_groups[analysis.content_hash].append(analysis.path)
        
        # Find groups with multiple files (duplicates)
        for hash_val, files in hash_groups.items():
            if len(files) > 1:
                self.duplicate_groups.append(files)
                logger.info(f"Found duplicate group: {files}")
    
    def get_optimization_plan(self) -> OptimizationPlan:
        """Generate optimization plan based on analysis"""
        files_to_remove = []
        files_to_rename = {}
        total_size_reduction = 0
        
        for analysis in self.file_analyses:
            if analysis.recommendation == 'remove':
                files_to_remove.append(analysis.path)
                total_size_reduction += analysis.file_size
            elif analysis.recommendation == 'rename':
                new_name = self._suggest_new_name(analysis.path, analysis.category)
                if new_name != analysis.path:
                    files_to_rename[analysis.path] = new_name
        
        # Calculate performance improvements
        performance_improvements = {}
        test_runners = [a for a in self.file_analyses if a.category == 'test_runner']
        if test_runners:
            kept_runners = [a for a in test_runners if a.recommendation == 'keep']
            if kept_runners:
                avg_score = sum(a.performance_score for a in kept_runners) / len(kept_runners)
                performance_improvements['test_runner_avg_score'] = avg_score
        
        return OptimizationPlan(
            files_to_remove=files_to_remove,
            files_to_rename=files_to_rename,
            total_files_analyzed=len(self.file_analyses),
            potential_size_reduction=total_size_reduction,
            performance_improvements=performance_improvements
        )
    
    def _suggest_new_name(self, old_path: str, category: str) -> str:
        """Suggest new standardized name for files"""
        path = Path(old_path)
        name = path.name
        
        if category == 'test_runner':
            if 'ultra_fast' in name:
                return str(path.parent / 'test_runner_fast.py')
            elif name == 'run_tests.py':
                return str(path.parent / 'test_runner_comprehensive.py')
        
        elif category == 'test':
            if 'sota' in name:
                new_name = name.replace('sota_', '').replace('_sota', '')
                return str(path.parent / new_name)
        
        return old_path
    
    def save_analysis_report(self, output_path: str = "codebase_analysis_report.json"):
        """Save detailed analysis report to JSON file"""
        report = {
            'analysis_summary': {
                'total_files': len(self.file_analyses),
                'categories': {},
                'recommendations': {},
                'duplicate_groups': len(self.duplicate_groups)
            },
            'file_analyses': [asdict(analysis) for analysis in self.file_analyses],
            'duplicate_groups': self.duplicate_groups,
            'optimization_plan': asdict(self.get_optimization_plan())
        }
        
        # Calculate category and recommendation counts
        for analysis in self.file_analyses:
            cat = analysis.category
            rec = analysis.recommendation
            
            report['analysis_summary']['categories'][cat] = report['analysis_summary']['categories'].get(cat, 0) + 1
            report['analysis_summary']['recommendations'][rec] = report['analysis_summary']['recommendations'].get(rec, 0) + 1
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Analysis report saved to {output_path}")
        return report

def main():
    """Main function for command-line usage"""
    analyzer = CodebaseAnalyzer()
    analyses = analyzer.analyze_codebase()
    
    # Print summary
    print(f"\n=== Codebase Analysis Summary ===")
    print(f"Total files analyzed: {len(analyses)}")
    
    categories = {}
    recommendations = {}
    
    for analysis in analyses:
        categories[analysis.category] = categories.get(analysis.category, 0) + 1
        recommendations[analysis.recommendation] = recommendations.get(analysis.recommendation, 0) + 1
    
    print(f"\nFile Categories:")
    for category, count in sorted(categories.items()):
        print(f"  {category}: {count}")
    
    print(f"\nRecommendations:")
    for rec, count in sorted(recommendations.items()):
        print(f"  {rec}: {count}")
    
    print(f"\nDuplicate groups found: {len(analyzer.duplicate_groups)}")
    
    # Save detailed report
    report = analyzer.save_analysis_report()
    
    # Show optimization plan
    plan = analyzer.get_optimization_plan()
    print(f"\n=== Optimization Plan ===")
    print(f"Files to remove: {len(plan.files_to_remove)}")
    print(f"Files to rename: {len(plan.files_to_rename)}")
    print(f"Potential size reduction: {plan.potential_size_reduction / 1024:.1f} KB")
    
    if plan.files_to_remove:
        print(f"\nFiles recommended for removal:")
        for file_path in plan.files_to_remove[:10]:  # Show first 10
            print(f"  - {file_path}")
        if len(plan.files_to_remove) > 10:
            print(f"  ... and {len(plan.files_to_remove) - 10} more")
    
    if plan.files_to_rename:
        print(f"\nFiles recommended for renaming:")
        for old, new in list(plan.files_to_rename.items())[:5]:  # Show first 5
            print(f"  - {old} -> {Path(new).name}")

if __name__ == "__main__":
    main()