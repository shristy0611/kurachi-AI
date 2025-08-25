#!/usr/bin/env python3
"""
Performance baseline capture utility for codebase optimization.
Captures current performance metrics to use as baseline for validation.
"""

import time
import json
import sys
import subprocess
from pathlib import Path
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

class PerformanceBaseline:
    """Captures performance baseline metrics."""
    
    def __init__(self):
        self.baseline_data = {
            "timestamp": time.strftime("%Y%m%d_%H%M%S"),
            "baseline_type": "pre_optimization",
            "metrics": {}
        }
    
    def capture_import_performance(self) -> Dict[str, float]:
        """Capture import performance for critical modules."""
        print("📊 Capturing import performance...")
        
        import_tests = [
            "services.intelligent_translation",
            "services.multilingual_conversation_interface",
            "services.document_processors",
            "services.language_detection",
            "services.translation_service",
            "services.preference_manager",
            "models.database",
            "utils.logger",
        ]
        
        import_times = {}
        
        for module_name in import_tests:
            try:
                start_time = time.time()
                __import__(module_name)
                import_time = time.time() - start_time
                import_times[f"{module_name}_import_time"] = import_time
                print(f"  ✅ {module_name}: {import_time:.4f}s")
            except ImportError as e:
                print(f"  ❌ {module_name}: Import failed - {e}")
                import_times[f"{module_name}_import_time"] = -1
            except Exception as e:
                print(f"  ❌ {module_name}: Error - {e}")
                import_times[f"{module_name}_import_time"] = -1
        
        return import_times
    
    def capture_test_runner_performance(self) -> Dict[str, float]:
        """Capture test runner performance."""
        print("🏃 Capturing test runner performance...")
        
        test_runners = [
            ("test_runner_fast.py", "fast_runner"),
            ("run_tests.py", "comprehensive_runner"),
        ]
        
        runner_times = {}
        
        for runner_file, metric_name in test_runners:
            if not Path(runner_file).exists():
                print(f"  ⚠️  {runner_file} not found, skipping")
                runner_times[f"{metric_name}_time"] = -1
                continue
            
            try:
                print(f"  🔄 Running {runner_file}...")
                start_time = time.time()
                result = subprocess.run(
                    [sys.executable, runner_file],
                    capture_output=True,
                    text=True,
                    timeout=60  # 1 minute timeout
                )
                duration = time.time() - start_time
                
                if result.returncode == 0:
                    runner_times[f"{metric_name}_time"] = duration
                    print(f"  ✅ {runner_file}: {duration:.3f}s")
                else:
                    runner_times[f"{metric_name}_time"] = -1
                    print(f"  ❌ {runner_file}: Failed with exit code {result.returncode}")
                    
            except subprocess.TimeoutExpired:
                runner_times[f"{metric_name}_time"] = -1
                print(f"  ❌ {runner_file}: Timed out after 60 seconds")
            except Exception as e:
                runner_times[f"{metric_name}_time"] = -1
                print(f"  ❌ {runner_file}: Error - {e}")
        
        return runner_times
    
    def capture_functionality_performance(self) -> Dict[str, float]:
        """Capture performance of key functionality."""
        print("⚡ Capturing functionality performance...")
        
        functionality_times = {}
        
        # Test language detection performance
        try:
            from services.language_detection import language_detection_service
            start_time = time.time()
            language_detection_service.detect_language("Hello world test")
            functionality_times["language_detection_time"] = time.time() - start_time
            print(f"  ✅ Language detection: {functionality_times['language_detection_time']:.4f}s")
        except Exception as e:
            functionality_times["language_detection_time"] = -1
            print(f"  ❌ Language detection: {e}")
        
        # Test translation service performance
        try:
            from services.translation_service import translation_service
            start_time = time.time()
            translation_service.translate("Hello", "en", "es")
            functionality_times["translation_service_time"] = time.time() - start_time
            print(f"  ✅ Translation service: {functionality_times['translation_service_time']:.4f}s")
        except Exception as e:
            functionality_times["translation_service_time"] = -1
            print(f"  ❌ Translation service: {e}")
        
        # Test document processor performance
        try:
            from services.document_processors import processor_factory
            start_time = time.time()
            processor = processor_factory.get_processor("txt")
            functionality_times["document_processor_time"] = time.time() - start_time
            print(f"  ✅ Document processor: {functionality_times['document_processor_time']:.4f}s")
        except Exception as e:
            functionality_times["document_processor_time"] = -1
            print(f"  ❌ Document processor: {e}")
        
        return functionality_times
    
    def capture_file_system_metrics(self) -> Dict[str, Any]:
        """Capture file system metrics."""
        print("📁 Capturing file system metrics...")
        
        metrics = {}
        
        # Count files by type
        file_counts = {
            "total_files": 0,
            "python_files": 0,
            "test_files": 0,
            "json_files": 0,
            "md_files": 0
        }
        
        for file_path in Path(".").rglob("*"):
            if file_path.is_file() and not any(part.startswith('.') for part in file_path.parts[1:]):
                file_counts["total_files"] += 1
                
                if file_path.suffix == ".py":
                    file_counts["python_files"] += 1
                    if "test" in file_path.name:
                        file_counts["test_files"] += 1
                elif file_path.suffix == ".json":
                    file_counts["json_files"] += 1
                elif file_path.suffix == ".md":
                    file_counts["md_files"] += 1
        
        metrics.update(file_counts)
        
        # Repository size (approximate)
        try:
            total_size = sum(f.stat().st_size for f in Path(".").rglob("*") if f.is_file())
            metrics["repository_size_bytes"] = total_size
            metrics["repository_size_mb"] = total_size / (1024 * 1024)
        except Exception as e:
            print(f"  ⚠️  Could not calculate repository size: {e}")
            metrics["repository_size_bytes"] = -1
            metrics["repository_size_mb"] = -1
        
        print(f"  📊 Total files: {file_counts['total_files']}")
        print(f"  🐍 Python files: {file_counts['python_files']}")
        print(f"  🧪 Test files: {file_counts['test_files']}")
        print(f"  📄 JSON files: {file_counts['json_files']}")
        print(f"  📝 Markdown files: {file_counts['md_files']}")
        if metrics["repository_size_mb"] > 0:
            print(f"  💾 Repository size: {metrics['repository_size_mb']:.1f} MB")
        
        return metrics
    
    def capture_full_baseline(self) -> Dict[str, Any]:
        """Capture complete performance baseline."""
        print("🚀 Capturing performance baseline...")
        print("=" * 50)
        
        start_time = time.time()
        
        # Capture all metrics
        import_metrics = self.capture_import_performance()
        runner_metrics = self.capture_test_runner_performance()
        functionality_metrics = self.capture_functionality_performance()
        filesystem_metrics = self.capture_file_system_metrics()
        
        # Combine all metrics
        self.baseline_data["metrics"].update(import_metrics)
        self.baseline_data["metrics"].update(runner_metrics)
        self.baseline_data["metrics"].update(functionality_metrics)
        self.baseline_data["metrics"].update(filesystem_metrics)
        
        # Add capture duration
        self.baseline_data["capture_duration"] = time.time() - start_time
        
        print("\n" + "=" * 50)
        print("📊 BASELINE CAPTURE SUMMARY")
        print("=" * 50)
        print(f"Metrics captured: {len(self.baseline_data['metrics'])}")
        print(f"Capture duration: {self.baseline_data['capture_duration']:.3f}s")
        
        # Show key metrics
        key_metrics = [
            ("fast_runner_time", "Fast test runner"),
            ("comprehensive_runner_time", "Comprehensive runner"),
            ("total_files", "Total files"),
            ("python_files", "Python files"),
            ("repository_size_mb", "Repository size (MB)")
        ]
        
        print("\n📈 Key Metrics:")
        for metric_key, description in key_metrics:
            if metric_key in self.baseline_data["metrics"]:
                value = self.baseline_data["metrics"][metric_key]
                if value >= 0:
                    if "time" in metric_key:
                        print(f"  {description}: {value:.3f}s")
                    elif "size_mb" in metric_key:
                        print(f"  {description}: {value:.1f} MB")
                    else:
                        print(f"  {description}: {value}")
                else:
                    print(f"  {description}: Not available")
        
        return self.baseline_data
    
    def save_baseline(self, filename: str = None) -> str:
        """Save baseline data to file."""
        if filename is None:
            filename = f"performance_baseline_{self.baseline_data['timestamp']}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.baseline_data, f, indent=2)
        
        print(f"\n💾 Baseline saved: {filename}")
        return filename


def main():
    """CLI interface for performance baseline capture."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Capture performance baseline for optimization")
    parser.add_argument("--output", "-o", help="Output filename for baseline data")
    parser.add_argument("--imports-only", action="store_true", help="Capture only import performance")
    parser.add_argument("--runners-only", action="store_true", help="Capture only test runner performance")
    parser.add_argument("--functionality-only", action="store_true", help="Capture only functionality performance")
    parser.add_argument("--filesystem-only", action="store_true", help="Capture only filesystem metrics")
    
    args = parser.parse_args()
    
    baseline = PerformanceBaseline()
    
    if args.imports_only:
        metrics = baseline.capture_import_performance()
        baseline.baseline_data["metrics"] = metrics
    elif args.runners_only:
        metrics = baseline.capture_test_runner_performance()
        baseline.baseline_data["metrics"] = metrics
    elif args.functionality_only:
        metrics = baseline.capture_functionality_performance()
        baseline.baseline_data["metrics"] = metrics
    elif args.filesystem_only:
        metrics = baseline.capture_file_system_metrics()
        baseline.baseline_data["metrics"] = metrics
    else:
        baseline.capture_full_baseline()
    
    # Save baseline
    filename = baseline.save_baseline(args.output)
    
    print(f"\n✅ Baseline capture complete!")
    print(f"Use this file for validation: --baseline {filename}")


if __name__ == "__main__":
    main()