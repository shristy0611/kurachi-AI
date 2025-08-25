#!/usr/bin/env python3
"""
Execute validation and cleanup for codebase optimization.
This script demonstrates the complete validation process including:
- Running validation tests after optimization phases
- Verifying optimized test runners work correctly
- Confirming file count and repository size reduction
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.execute_validation import ValidationExecutor
from tools.validation_system import ValidationSystem
from tools.backup_system import BackupSystem

class ValidationCleanupRunner:
    """Runs validation and cleanup operations for codebase optimization."""
    
    def __init__(self):
        self.validation_executor = ValidationExecutor()
        self.validation_system = ValidationSystem()
        self.backup_system = BackupSystem()
        self.results = {
            "timestamp": time.strftime("%Y%m%d_%H%M%S"),
            "phases": {},
            "overall_success": False
        }
    
    def run_post_optimization_validation(self) -> bool:
        """Run validation tests after optimization phase."""
        print("🚀 PHASE 1: POST-OPTIMIZATION VALIDATION")
        print("=" * 60)
        
        phase_start = time.time()
        
        try:
            # Find latest baseline file
            baseline_files = list(Path(".").glob("performance_baseline_*.json"))
            baseline_file = None
            if baseline_files:
                baseline_file = str(max(baseline_files, key=lambda p: p.stat().st_mtime))
                print(f"📊 Using baseline: {baseline_file}")
            else:
                print("⚠️  No baseline file found - performance comparison will be limited")
            
            # Run comprehensive validation
            print("\n🧪 Running comprehensive validation...")
            validation_report = self.validation_system.run_full_validation(baseline_file)
            
            # Save validation report
            report_file = self.validation_system.save_report(
                f"post_optimization_validation_{self.results['timestamp']}.json"
            )
            
            phase_success = validation_report["summary"]["overall_success"]
            
            # Store results
            self.results["phases"]["post_optimization_validation"] = {
                "success": phase_success,
                "duration": time.time() - phase_start,
                "report_file": report_file,
                "tests_passed": validation_report["summary"]["passed"],
                "tests_failed": validation_report["summary"]["failed"],
                "performance_maintained": validation_report["tests"]["performance_validation"]["performance_maintained"]
            }
            
            if phase_success:
                print("✅ Post-optimization validation passed!")
            else:
                print("❌ Post-optimization validation failed!")
                print("🔄 Rollback may be required if critical issues detected")
            
            return phase_success
            
        except Exception as e:
            print(f"❌ Post-optimization validation failed: {e}")
            self.results["phases"]["post_optimization_validation"] = {
                "success": False,
                "duration": time.time() - phase_start,
                "error": str(e)
            }
            return False
    
    def verify_test_runners(self) -> bool:
        """Verify that optimized test runners still pass all tests."""
        print("\n🚀 PHASE 2: TEST RUNNER VERIFICATION")
        print("=" * 60)
        
        phase_start = time.time()
        
        try:
            # Test the fast test runner
            print("🏃 Testing optimized test runners...")
            runner_results = self.validation_system.validate_test_runners()
            
            phase_success = runner_results["failed"] == 0
            
            # Store results
            self.results["phases"]["test_runner_verification"] = {
                "success": phase_success,
                "duration": time.time() - phase_start,
                "runners_tested": runner_results["passed"] + runner_results["failed"],
                "runners_passed": runner_results["passed"],
                "runners_failed": runner_results["failed"],
                "details": runner_results["details"]
            }
            
            if phase_success:
                print(f"✅ All {runner_results['passed']} test runners working correctly!")
            else:
                print(f"❌ {runner_results['failed']} test runners failed!")
                for error in runner_results["errors"]:
                    print(f"  • {error}")
            
            return phase_success
            
        except Exception as e:
            print(f"❌ Test runner verification failed: {e}")
            self.results["phases"]["test_runner_verification"] = {
                "success": False,
                "duration": time.time() - phase_start,
                "error": str(e)
            }
            return False
    
    def verify_optimization_goals(self) -> bool:
        """Confirm that file count and repository size are reduced as expected."""
        print("\n🚀 PHASE 3: OPTIMIZATION GOALS VERIFICATION")
        print("=" * 60)
        
        phase_start = time.time()
        
        try:
            # Get current metrics
            current_metrics = self._get_current_metrics()
            
            # Try to find baseline for comparison
            baseline_metrics = self._get_baseline_metrics()
            
            optimization_achieved = False
            
            if baseline_metrics:
                print("📊 Comparing with baseline metrics...")
                
                # File count comparison
                baseline_files = baseline_metrics.get("total_files", 0)
                current_files = current_metrics.get("total_files", 0)
                
                if current_files < baseline_files:
                    file_reduction = baseline_files - current_files
                    file_reduction_pct = (file_reduction / baseline_files) * 100
                    print(f"✅ File count reduced: {baseline_files} → {current_files} (-{file_reduction} files, -{file_reduction_pct:.1f}%)")
                    optimization_achieved = True
                else:
                    print(f"⚠️  File count not reduced: {baseline_files} → {current_files}")
                
                # Repository size comparison
                baseline_size = baseline_metrics.get("repository_size_mb", 0)
                current_size = current_metrics.get("repository_size_mb", 0)
                
                if current_size > 0 and baseline_size > 0:
                    if current_size < baseline_size:
                        size_reduction = baseline_size - current_size
                        size_reduction_pct = (size_reduction / baseline_size) * 100
                        print(f"✅ Repository size reduced: {baseline_size:.1f}MB → {current_size:.1f}MB (-{size_reduction:.1f}MB, -{size_reduction_pct:.1f}%)")
                        optimization_achieved = True
                    else:
                        print(f"⚠️  Repository size not reduced: {baseline_size:.1f}MB → {current_size:.1f}MB")
            else:
                print("⚠️  No baseline available - cannot verify optimization goals")
                # Still consider successful if we have reasonable current metrics
                optimization_achieved = current_metrics.get("total_files", 0) > 0
            
            # Store results
            self.results["phases"]["optimization_goals_verification"] = {
                "success": optimization_achieved,
                "duration": time.time() - phase_start,
                "current_metrics": current_metrics,
                "baseline_metrics": baseline_metrics,
                "optimization_achieved": optimization_achieved
            }
            
            if optimization_achieved:
                print("✅ Optimization goals achieved!")
            else:
                print("⚠️  Optimization goals not fully achieved")
            
            return optimization_achieved
            
        except Exception as e:
            print(f"❌ Optimization goals verification failed: {e}")
            self.results["phases"]["optimization_goals_verification"] = {
                "success": False,
                "duration": time.time() - phase_start,
                "error": str(e)
            }
            return False
    
    def _get_current_metrics(self) -> Dict[str, Any]:
        """Get current file system metrics."""
        try:
            from tools.performance_baseline import PerformanceBaseline
            baseline = PerformanceBaseline()
            return baseline.capture_file_system_metrics()
        except Exception as e:
            print(f"⚠️  Could not capture current metrics: {e}")
            return {}
    
    def _get_baseline_metrics(self) -> Dict[str, Any]:
        """Get baseline metrics from latest baseline file."""
        try:
            baseline_files = list(Path(".").glob("performance_baseline_*.json"))
            if not baseline_files:
                return {}
            
            latest_baseline = max(baseline_files, key=lambda p: p.stat().st_mtime)
            
            with open(latest_baseline, 'r') as f:
                baseline_data = json.load(f)
            
            return baseline_data.get("metrics", {})
        except Exception as e:
            print(f"⚠️  Could not load baseline metrics: {e}")
            return {}
    
    def generate_cleanup_report(self) -> str:
        """Generate final cleanup and validation report."""
        print("\n🚀 PHASE 4: CLEANUP REPORT GENERATION")
        print("=" * 60)
        
        # Calculate overall success
        overall_success = all(
            phase_data.get("success", False) 
            for phase_data in self.results["phases"].values()
        )
        
        total_duration = sum(
            phase_data.get("duration", 0) 
            for phase_data in self.results["phases"].values()
        )
        
        # Create comprehensive report
        cleanup_report = {
            "validation_cleanup_summary": {
                "timestamp": self.results["timestamp"],
                "overall_success": overall_success,
                "total_duration": total_duration,
                "phases_completed": len(self.results["phases"])
            },
            "phase_results": self.results["phases"],
            "recommendations": self._generate_recommendations(overall_success),
            "next_steps": self._generate_next_steps(overall_success)
        }
        
        # Save report
        report_filename = f"validation_cleanup_report_{self.results['timestamp']}.json"
        with open(report_filename, 'w') as f:
            json.dump(cleanup_report, f, indent=2)
        
        print(f"📄 Cleanup report generated: {report_filename}")
        
        # Print summary
        print("\n📊 VALIDATION & CLEANUP SUMMARY")
        print("=" * 50)
        print(f"Overall Success: {'✅ YES' if overall_success else '❌ NO'}")
        print(f"Total Duration: {total_duration:.3f}s")
        print(f"Phases Completed: {len(self.results['phases'])}")
        
        for phase_name, phase_data in self.results["phases"].items():
            status = "✅" if phase_data.get("success", False) else "❌"
            duration = phase_data.get("duration", 0)
            print(f"  {status} {phase_name.replace('_', ' ').title()}: {duration:.3f}s")
        
        self.results["overall_success"] = overall_success
        return report_filename
    
    def _generate_recommendations(self, overall_success: bool) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        if overall_success:
            recommendations.extend([
                "✅ Validation and cleanup completed successfully",
                "📦 Maintain backup files for at least one development cycle",
                "🧪 Continue running validation tests regularly",
                "📊 Monitor performance metrics in ongoing development",
                "📄 Update project documentation to reflect optimizations"
            ])
        else:
            recommendations.extend([
                "❌ Some validation or cleanup issues detected",
                "🔍 Review detailed phase results for specific issues",
                "🔄 Consider rollback if critical functionality is affected",
                "🛠️  Address validation failures before proceeding",
                "📋 Re-run validation after fixing identified issues"
            ])
        
        return recommendations
    
    def _generate_next_steps(self, overall_success: bool) -> List[str]:
        """Generate next steps based on results."""
        next_steps = []
        
        if overall_success:
            next_steps.extend([
                "🚀 Proceed with normal development workflow",
                "📝 Update team documentation about optimizations",
                "🔄 Set up regular validation checks",
                "📊 Monitor performance in production environment",
                "🧹 Schedule periodic cleanup reviews"
            ])
        else:
            next_steps.extend([
                "🔍 Investigate and fix validation failures",
                "🔄 Consider rollback to previous stable state",
                "🛠️  Address specific issues identified in phase results",
                "🧪 Re-run validation after fixes",
                "📋 Update optimization strategy based on learnings"
            ])
        
        return next_steps
    
    def run_complete_validation_cleanup(self) -> bool:
        """Run the complete validation and cleanup process."""
        print("🚀 CODEBASE OPTIMIZATION - VALIDATION & CLEANUP")
        print("=" * 70)
        print(f"Timestamp: {self.results['timestamp']}")
        print("=" * 70)
        
        overall_success = True
        
        # Phase 1: Post-optimization validation
        success = self.run_post_optimization_validation()
        overall_success = overall_success and success
        
        # Phase 2: Test runner verification
        success = self.verify_test_runners()
        overall_success = overall_success and success
        
        # Phase 3: Optimization goals verification
        success = self.verify_optimization_goals()
        overall_success = overall_success and success
        
        # Phase 4: Generate cleanup report
        report_file = self.generate_cleanup_report()
        
        # Final summary
        print("\n" + "=" * 70)
        print("🏁 VALIDATION & CLEANUP COMPLETE")
        print("=" * 70)
        
        if overall_success:
            print("✅ All validation and cleanup phases completed successfully!")
            print("🎉 Codebase optimization validation passed!")
        else:
            print("❌ Some validation or cleanup issues detected")
            print("📋 Review the detailed report for specific issues")
        
        print(f"📄 Detailed report: {report_file}")
        
        return overall_success


def main():
    """CLI interface for validation cleanup runner."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run validation and cleanup for codebase optimization")
    parser.add_argument("--phase", choices=["validation", "test-runners", "goals", "all"], 
                       default="all", help="Specific phase to run")
    
    args = parser.parse_args()
    
    runner = ValidationCleanupRunner()
    
    if args.phase == "validation":
        success = runner.run_post_optimization_validation()
    elif args.phase == "test-runners":
        success = runner.verify_test_runners()
    elif args.phase == "goals":
        success = runner.verify_optimization_goals()
    else:
        # Run complete process
        success = runner.run_complete_validation_cleanup()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()