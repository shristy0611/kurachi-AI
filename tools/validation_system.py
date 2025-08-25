#!/usr/bin/env python3
"""
Comprehensive validation system for codebase optimization.
Validates imports, functionality, and performance after optimization changes.
"""

import os
import sys
import time
import json
import importlib
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import traceback

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

class ValidationSystem:
    """Comprehensive validation system for optimization changes."""
    
    def __init__(self):
        self.results = []
        self.start_time = time.time()
        self.baseline_performance = {}
        self.validation_report = {
            "timestamp": time.strftime("%Y%m%d_%H%M%S"),
            "validation_type": "codebase_optimization",
            "tests": {},
            "summary": {}
        }
    
    def validate_imports(self) -> Dict[str, Any]:
        """Validate all critical imports work after renaming."""
        print("🔍 Validating imports...")
        
        import_tests = {
            # Core services
            "services.intelligent_translation": "IntelligentTranslationService",
            "services.multilingual_conversation_interface": "MultilingualConversationInterface", 
            "services.document_processors": "processor_factory",
            "services.language_detection": "language_detection_service",
            "services.translation_service": "translation_service",
            "services.preference_manager": "preference_manager",
            "services.document_service": "DocumentService",
            "services.chat_service": "ChatService",
            
            # Models
            "models.database": "db_manager",
            
            # Utils
            "utils.logger": None,  # Module import only
            "utils.helpers": None,  # Module import only
            
            # Test runners (after optimization)
            "test_runner_fast": "UltraFastTestAnalyzer",
        }
        
        results = {"passed": 0, "failed": 0, "errors": []}
        
        for module_name, class_name in import_tests.items():
            try:
                module = importlib.import_module(module_name)
                
                if class_name:
                    if hasattr(module, class_name):
                        print(f"  ✅ {module_name}.{class_name}")
                        results["passed"] += 1
                    else:
                        error_msg = f"Class {class_name} not found in {module_name}"
                        print(f"  ❌ {error_msg}")
                        results["failed"] += 1
                        results["errors"].append(error_msg)
                else:
                    print(f"  ✅ {module_name}")
                    results["passed"] += 1
                    
            except ImportError as e:
                error_msg = f"Import failed for {module_name}: {str(e)}"
                print(f"  ❌ {error_msg}")
                results["failed"] += 1
                results["errors"].append(error_msg)
            except Exception as e:
                error_msg = f"Unexpected error importing {module_name}: {str(e)}"
                print(f"  ❌ {error_msg}")
                results["failed"] += 1
                results["errors"].append(error_msg)
        
        self.validation_report["tests"]["import_validation"] = results
        return results
    
    def validate_functionality(self) -> Dict[str, Any]:
        """Validate core functionality still works after optimization."""
        print("🧪 Validating functionality...")
        
        functionality_tests = [
            ("Language Detection", self._test_language_detection),
            ("Translation Service", self._test_translation_service),
            ("Document Processing", self._test_document_processing),
            ("Database Connection", self._test_database_connection),
            ("Preference Management", self._test_preference_management),
            ("Multilingual Interface", self._test_multilingual_interface),
            ("Configuration System", self._test_configuration_system),
        ]
        
        results = {"passed": 0, "failed": 0, "errors": [], "details": {}}
        
        for test_name, test_func in functionality_tests:
            try:
                start_time = time.time()
                success, details = test_func()
                duration = time.time() - start_time
                
                if success:
                    print(f"  ✅ {test_name} ({duration:.3f}s)")
                    results["passed"] += 1
                else:
                    print(f"  ❌ {test_name} ({duration:.3f}s): {details}")
                    results["failed"] += 1
                    results["errors"].append(f"{test_name}: {details}")
                
                results["details"][test_name] = {
                    "success": success,
                    "duration": duration,
                    "details": details
                }
                
            except Exception as e:
                error_msg = f"{test_name}: {str(e)}"
                print(f"  ❌ {error_msg}")
                results["failed"] += 1
                results["errors"].append(error_msg)
                results["details"][test_name] = {
                    "success": False,
                    "duration": 0,
                    "details": str(e)
                }
        
        self.validation_report["tests"]["functionality_validation"] = results
        return results
    
    def validate_performance(self, baseline_file: Optional[str] = None) -> Dict[str, Any]:
        """Validate performance hasn't regressed after optimization."""
        print("⚡ Validating performance...")
        
        # Load baseline if provided
        if baseline_file and Path(baseline_file).exists():
            with open(baseline_file, 'r') as f:
                self.baseline_performance = json.load(f)
        
        # Run performance tests
        performance_results = self._run_performance_tests()
        
        # Compare with baseline
        comparison = self._compare_performance(performance_results)
        
        results = {
            "current_performance": performance_results,
            "baseline_comparison": comparison,
            "performance_maintained": comparison.get("overall_status", "unknown") != "regression"
        }
        
        self.validation_report["tests"]["performance_validation"] = results
        return results
    
    def validate_test_runners(self) -> Dict[str, Any]:
        """Validate that optimized test runners work correctly."""
        print("🏃 Validating test runners...")
        
        test_runners = [
            ("test_runner_fast.py", "Fast test runner"),
            ("run_tests.py", "Comprehensive test runner") if Path("run_tests.py").exists() else None,
            ("test_runner_comprehensive.py", "Renamed comprehensive runner") if Path("test_runner_comprehensive.py").exists() else None
        ]
        
        # Filter out None entries
        test_runners = [tr for tr in test_runners if tr is not None]
        
        results = {"passed": 0, "failed": 0, "errors": [], "details": {}}
        
        for runner_file, description in test_runners:
            if not Path(runner_file).exists():
                error_msg = f"{description} not found: {runner_file}"
                print(f"  ❌ {error_msg}")
                results["failed"] += 1
                results["errors"].append(error_msg)
                continue
            
            try:
                # Test that the runner can be imported and executed
                start_time = time.time()
                result = subprocess.run(
                    [sys.executable, runner_file],
                    capture_output=True,
                    text=True,
                    timeout=30  # 30 second timeout
                )
                duration = time.time() - start_time
                
                if result.returncode == 0:
                    print(f"  ✅ {description} ({duration:.3f}s)")
                    results["passed"] += 1
                    results["details"][runner_file] = {
                        "success": True,
                        "duration": duration,
                        "output_lines": len(result.stdout.split('\n'))
                    }
                else:
                    error_msg = f"{description} failed with exit code {result.returncode}"
                    print(f"  ❌ {error_msg}")
                    results["failed"] += 1
                    results["errors"].append(f"{error_msg}: {result.stderr[:200]}")
                    results["details"][runner_file] = {
                        "success": False,
                        "duration": duration,
                        "error": result.stderr[:200]
                    }
                    
            except subprocess.TimeoutExpired:
                error_msg = f"{description} timed out after 30 seconds"
                print(f"  ❌ {error_msg}")
                results["failed"] += 1
                results["errors"].append(error_msg)
            except Exception as e:
                error_msg = f"{description} error: {str(e)}"
                print(f"  ❌ {error_msg}")
                results["failed"] += 1
                results["errors"].append(error_msg)
        
        self.validation_report["tests"]["test_runner_validation"] = results
        return results
    
    def _test_language_detection(self) -> Tuple[bool, str]:
        """Test language detection functionality."""
        try:
            from services.language_detection import language_detection_service
            
            # Test basic detection
            result = language_detection_service.detect_language("Hello world")
            if result and isinstance(result, str) and result != "unknown":
                return True, f"Detected: {result}"
            else:
                return False, "No language detected"
                
        except Exception as e:
            return False, str(e)
    
    def _test_translation_service(self) -> Tuple[bool, str]:
        """Test translation service functionality."""
        try:
            from services.translation_service import translation_service
            
            # Test basic translation capability
            result = translation_service.translate("Hello", "en", "es")
            if result and isinstance(result, dict) and "translated_text" in result:
                translated_text = result["translated_text"]
                return True, f"Translation successful: {translated_text[:50] if translated_text else 'empty'}"
            else:
                return False, "Translation returned empty result"
                
        except Exception as e:
            return False, str(e)
    
    def _test_document_processing(self) -> Tuple[bool, str]:
        """Test document processing functionality."""
        try:
            from services.document_processors import processor_factory
            
            # Test processor factory with a proper filename
            processor = processor_factory.get_processor("test.txt")
            if processor:
                return True, f"Processor found: {type(processor).__name__}"
            else:
                return False, "No processor found for txt files"
                
        except Exception as e:
            return False, str(e)
    
    def _test_database_connection(self) -> Tuple[bool, str]:
        """Test database connection functionality."""
        try:
            from models.database import db_manager
            
            # Test database connection
            if hasattr(db_manager, 'get_connection'):
                return True, "Database manager accessible"
            else:
                return False, "Database manager missing get_connection method"
                
        except Exception as e:
            return False, str(e)
    
    def _test_preference_management(self) -> Tuple[bool, str]:
        """Test preference management functionality."""
        try:
            from services.preference_manager import preference_manager
            
            # Test preference access
            if hasattr(preference_manager, 'get_preference'):
                return True, "Preference manager accessible"
            else:
                return False, "Preference manager missing get_preference method"
                
        except Exception as e:
            return False, str(e)
    
    def _test_multilingual_interface(self) -> Tuple[bool, str]:
        """Test multilingual interface functionality."""
        try:
            from services.multilingual_conversation_interface import MultilingualConversationInterface
            
            # Test interface creation
            interface = MultilingualConversationInterface()
            if interface:
                return True, "Multilingual interface created successfully"
            else:
                return False, "Failed to create multilingual interface"
                
        except Exception as e:
            return False, str(e)
    
    def _test_configuration_system(self) -> Tuple[bool, str]:
        """Test configuration system functionality."""
        try:
            import config
            
            # Test config access
            if hasattr(config, 'ai') or hasattr(config, 'database'):
                return True, "Configuration system accessible"
            else:
                return False, "Configuration system missing expected attributes"
                
        except Exception as e:
            return False, str(e)
    
    def _run_performance_tests(self) -> Dict[str, Any]:
        """Run performance tests and collect metrics."""
        performance_data = {}
        
        # Test import performance
        start_time = time.time()
        try:
            from services.intelligent_translation import IntelligentTranslationService
            performance_data["import_time"] = time.time() - start_time
        except:
            performance_data["import_time"] = -1
        
        # Test test runner performance
        if Path("test_runner_fast.py").exists():
            try:
                start_time = time.time()
                result = subprocess.run(
                    [sys.executable, "test_runner_fast.py"],
                    capture_output=True,
                    text=True,
                    timeout=15
                )
                if result.returncode == 0:
                    performance_data["test_runner_time"] = time.time() - start_time
                else:
                    performance_data["test_runner_time"] = -1
            except:
                performance_data["test_runner_time"] = -1
        
        return performance_data
    
    def _compare_performance(self, current: Dict[str, Any]) -> Dict[str, Any]:
        """Compare current performance with baseline."""
        if not self.baseline_performance:
            return {"status": "no_baseline", "message": "No baseline performance data available"}
        
        comparison = {"improvements": [], "regressions": [], "unchanged": []}
        
        for metric, current_value in current.items():
            if metric in self.baseline_performance:
                baseline_value = self.baseline_performance[metric]
                
                if current_value < 0 or baseline_value < 0:
                    comparison["unchanged"].append(f"{metric}: measurement failed")
                    continue
                
                # For time metrics, lower is better
                if "time" in metric:
                    if current_value < baseline_value * 0.9:  # 10% improvement threshold
                        improvement = ((baseline_value - current_value) / baseline_value) * 100
                        comparison["improvements"].append(f"{metric}: {improvement:.1f}% faster")
                    elif current_value > baseline_value * 1.1:  # 10% regression threshold
                        regression = ((current_value - baseline_value) / baseline_value) * 100
                        comparison["regressions"].append(f"{metric}: {regression:.1f}% slower")
                    else:
                        comparison["unchanged"].append(f"{metric}: similar performance")
        
        # Determine overall status
        if comparison["regressions"]:
            comparison["overall_status"] = "regression"
        elif comparison["improvements"]:
            comparison["overall_status"] = "improvement"
        else:
            comparison["overall_status"] = "stable"
        
        return comparison
    
    def run_full_validation(self, baseline_file: Optional[str] = None) -> Dict[str, Any]:
        """Run complete validation suite."""
        print("🚀 Starting comprehensive validation...")
        print("=" * 50)
        
        # Run all validation tests
        import_results = self.validate_imports()
        functionality_results = self.validate_functionality()
        performance_results = self.validate_performance(baseline_file)
        test_runner_results = self.validate_test_runners()
        
        # Calculate overall results
        total_passed = (import_results["passed"] + functionality_results["passed"] + 
                       test_runner_results["passed"])
        total_failed = (import_results["failed"] + functionality_results["failed"] + 
                       test_runner_results["failed"])
        
        overall_success = total_failed == 0 and performance_results["performance_maintained"]
        
        # Update summary
        self.validation_report["summary"] = {
            "total_tests": total_passed + total_failed,
            "passed": total_passed,
            "failed": total_failed,
            "success_rate": (total_passed / (total_passed + total_failed)) * 100 if (total_passed + total_failed) > 0 else 0,
            "performance_maintained": performance_results["performance_maintained"],
            "overall_success": overall_success,
            "duration": time.time() - self.start_time
        }
        
        # Print summary
        print("\n" + "=" * 50)
        print("📊 VALIDATION SUMMARY")
        print("=" * 50)
        print(f"Total Tests: {total_passed + total_failed}")
        print(f"Passed: {total_passed}")
        print(f"Failed: {total_failed}")
        print(f"Success Rate: {self.validation_report['summary']['success_rate']:.1f}%")
        print(f"Performance: {'✅ Maintained' if performance_results['performance_maintained'] else '❌ Regressed'}")
        print(f"Overall: {'✅ SUCCESS' if overall_success else '❌ FAILED'}")
        print(f"Duration: {self.validation_report['summary']['duration']:.3f}s")
        
        if total_failed > 0:
            print("\n❌ FAILURES:")
            for test_type, results in [
                ("Import", import_results),
                ("Functionality", functionality_results), 
                ("Test Runner", test_runner_results)
            ]:
                if results["errors"]:
                    print(f"\n{test_type} Errors:")
                    for error in results["errors"]:
                        print(f"  • {error}")
        
        return self.validation_report
    
    def save_report(self, filename: Optional[str] = None) -> str:
        """Save validation report to file."""
        if filename is None:
            filename = f"validation_report_{self.validation_report['timestamp']}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.validation_report, f, indent=2)
        
        print(f"📄 Validation report saved: {filename}")
        return filename


def main():
    """CLI interface for validation system."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Codebase optimization validation system")
    parser.add_argument("--baseline", help="Baseline performance file for comparison")
    parser.add_argument("--save-report", help="Save detailed report to specified file")
    parser.add_argument("--imports-only", action="store_true", help="Run only import validation")
    parser.add_argument("--functionality-only", action="store_true", help="Run only functionality validation")
    parser.add_argument("--performance-only", action="store_true", help="Run only performance validation")
    parser.add_argument("--test-runners-only", action="store_true", help="Run only test runner validation")
    
    args = parser.parse_args()
    
    validator = ValidationSystem()
    
    if args.imports_only:
        validator.validate_imports()
    elif args.functionality_only:
        validator.validate_functionality()
    elif args.performance_only:
        validator.validate_performance(args.baseline)
    elif args.test_runners_only:
        validator.validate_test_runners()
    else:
        # Run full validation
        report = validator.run_full_validation(args.baseline)
        
        # Save report if requested
        if args.save_report:
            validator.save_report(args.save_report)
        
        # Exit with appropriate code
        sys.exit(0 if report["summary"]["overall_success"] else 1)


if __name__ == "__main__":
    main()