# Codebase Analyzer

A comprehensive tool for analyzing codebases to identify optimization opportunities, categorize files, detect duplicates, and generate performance-based recommendations.

## Features

### File Analysis and Categorization
- **Test Runners**: Identifies and scores test runner scripts based on performance
- **Analysis Files**: Categorizes analysis and summary files
- **Test Files**: Identifies unit and integration tests
- **Source Code**: Categorizes Python source files
- **Configuration**: Identifies config files (YAML, JSON, etc.)
- **Documentation**: Identifies markdown and documentation files

### Duplicate Detection
- Content-based duplicate detection using MD5 hashing
- Similar file identification based on naming patterns and functionality
- Groups duplicate files for optimization recommendations

### Performance Scoring System
- Loads performance data from existing test analysis files
- Scores test runners based on execution time (lower time = higher score)
- Uses naming patterns as fallback scoring mechanism
- Prioritizes "ultra_fast" and optimized versions

### Optimization Recommendations
- **Keep**: High-performing, essential files
- **Remove**: Duplicate or low-performing files
- **Rename**: Files with non-standard naming conventions

## Usage

### Command Line
```bash
python tools/codebase_analyzer.py
```

### Programmatic Usage
```python
from tools.codebase_analyzer import CodebaseAnalyzer

# Initialize analyzer
analyzer = CodebaseAnalyzer()

# Analyze codebase
analyses = analyzer.analyze_codebase()

# Get optimization plan
plan = analyzer.get_optimization_plan()

# Save detailed report
report = analyzer.save_analysis_report("my_analysis.json")
```

## Output

### Console Output
- Summary of files analyzed by category
- Recommendation counts (keep/remove/rename)
- Optimization plan with file counts and size reduction
- Top files recommended for removal/renaming

### Analysis Report (JSON)
- Detailed analysis for each file including:
  - File path and category
  - Performance score
  - Dependencies (for Python files)
  - Similar files
  - Recommendation with reasoning
  - File metadata (size, modification time, content hash)
- Optimization plan with specific file lists
- Summary statistics

## Performance Scoring

### Test Runners
- Based on actual execution time from analysis files
- Fallback scoring based on naming patterns:
  - `ultra_fast`: 9.5
  - `fast`: 8.0
  - `optimized`: 7.0
  - `complete`: 6.0
  - `minimal`: 5.0

### Analysis Files
- Prefers comprehensive summaries: 8.0
- Recent files get higher scores: 6.0-7.0
- Older files get lower scores: 3.0

## Requirements Satisfied

### Requirement 1.1: File Categorization
✅ Identifies and categorizes files into test_runner, analysis, test, source, config, docs

### Requirement 1.2: Duplicate Detection
✅ Detects duplicate files using content hashing and similarity analysis

### Performance Scoring
✅ Implements scoring system based on test results data and naming patterns

## Example Results

```
=== Codebase Analysis Summary ===
Total files analyzed: 100

File Categories:
  analysis: 7
  source: 60
  test: 26
  test_runner: 7

Recommendations:
  keep: 95
  remove: 4
  rename: 1

=== Optimization Plan ===
Files to remove: 4
Files to rename: 1
Potential size reduction: 40.5 KB

Files recommended for removal:
  - run_tests.py
  - run_optimized_complete_tests.py
  - run_complete_tests_with_analysis.py
  - run_minimal_tests.py

Files recommended for renaming:
  - tests/unit/test_sota_markdown_chunking.py -> test_markdown_chunking.py ✅
```

## Testing

Run the test suite to validate functionality:
```bash
python tools/test_codebase_analyzer.py
```

The test suite validates:
- Performance data loading
- File categorization accuracy
- Test runner identification and scoring
- Duplicate detection
- Recommendation generation
- Report generation