#!/usr/bin/env python3
"""
Execute comprehensive validation and cleanup for codebase optimization.
Orchestrates the complete validation process with backup and rollback capabilities.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.backup_system import BackupSystem
from tools.validation_system import ValidationSystem
from tools.performance_baseline import PerformanceBaseline

class ValidationExecutor:
    """Orchestrates the complete validation and cleanup process."""
    
    def __init__(self):
        self.backup_system = BackupSystem()
        self.validation_system = ValidationSystem()
        self.performance_baseline = PerformanceBaseline()
        self.execution_log = {
            "timestamp": time.strftime("%Y%m%d_%H%M%S"),
            "phases": {},
            "overall_success": False,
            "backup_created": None,
            "baseline_captured": None,
            "validation_report": None
        }
    
    def execute_pre_optimization_phase(self) -> bool:
        """Execute pre-optimization validation phase."""
        print("🚀 PHASE 1: PRE-OPTIMIZATION VALIDATION")
        print("=" * 60)
        
        phase_start = time.time()
        phase_success = True
        
        try:
            # Step 1: Create backup
            print("\n📦 Step 1: Creating backup...")
            backup_path = self.backup_system.create_backup()
            self.execution_log["backup_created"] = backup_path
            print(f"✅ Backup created: {backup_path}")
            
            # Step 2: Capture performance baseline
            print("\n📊 Step 2: Capturing performance baseline...")
            baseline_data = self.performance_baseline.capture_full_baseline()
            baseline_file = self.performance_baseline.save_baseline()
            self.execution_log["baseline_captured"] = baseline_file
            print(f"✅ Baseline captured: {baseline_file}")
            
            # Step 3: Run initial validation
            print("\n🧪 Step 3: Running initial validation...")
            validation_report = self.validation_system.run_full_validation(baseline_file)
            
            if not validation_report["summary"]["overall_success"]:
                print("❌ Initial validation failed - system not ready for optimization")
                phase_success = False
            else:
                print("✅ Initial validation passed - system ready for optimization")
            
        except Exception as e:
            print(f"❌ Pre-optimization phase failed: {e}")
            phase_success = False
        
        phase_duration = time.time() - phase_start
        self.execution_log["phases"]["pre_optimization"] = {
            "success": phase_success,
            "duration": phase_duration,
            "backup_path": self.execution_log.get("backup_created"),
            "baseline_file": self.execution_log.get("baseline_captured")
        }
        
        print(f"\n📋 Phase 1 {'✅ COMPLETED' if phase_success else '❌ FAILED'} ({phase_duration:.3f}s)")
        return phase_success
    
    def execute_post_optimization_phase(self, baseline_file: Optional[str] = None) -> bool:
        """Execute post-optimization validation phase."""
        print("\n🚀 PHASE 2: POST-OPTIMIZATION VALIDATION")
        print("=" * 60)
        
        phase_start = time.time()
        phase_success = True
        
        try:
            # Use provided baseline or find the latest one
            if baseline_file is None:
                baseline_file = self.execution_log.get("baseline_captured")
                if baseline_file is None:
                    # Try to find latest baseline file
                    baseline_files = list(Path(".").glob("performance_baseline_*.json"))
                    if baseline_files:
                        baseline_file = str(max(baseline_files, key=lambda p: p.stat().st_mtime))
                        print(f"📊 Using latest baseline: {baseline_file}")
                    else:
                        print("⚠️  No baseline file found - performance comparison will be limited")
            
            # Step 1: Run comprehensive validation
            print("\n🧪 Step 1: Running post-optimization validation...")
            validation_report = self.validation_system.run_full_validation(baseline_file)
            
            # Save validation report
            report_file = self.validation_system.save_report()
            self.execution_log["validation_report"] = report_file
            
            # Step 2: Check if rollback is needed
            if not validation_report["summary"]["overall_success"]:
                print("\n❌ Post-optimization validation failed!")
                
                # Offer rollback
                if self.execution_log.get("backup_created"):
                    print(f"🔄 Rollback available from: {self.execution_log['backup_created']}")
                    
                    # Auto-rollback if critical failures
                    critical_failures = validation_report["summary"]["failed"] > 5
                    if critical_failures:
                        print("🚨 Critical failures detected - initiating automatic rollback...")
                        rollback_success = self.backup_system.rollback(self.execution_log["backup_created"])
                        if rollback_success:
                            print("✅ Automatic rollback completed successfully")
                        else:
                            print("❌ Automatic rollback failed - manual intervention required")
                    else:
                        print("⚠️  Non-critical failures - rollback available but not automatic")
                
                phase_success = False
            else:
                print("\n✅ Post-optimization validation passed!")
                
                # Step 3: Verify optimizations achieved expected results
                print("\n📈 Step 3: Verifying optimization results...")
                optimization_success = self._verify_optimization_results(validation_report)
                
                if optimization_success:
                    print("✅ Optimization goals achieved")
                else:
                    print("⚠️  Optimization goals partially achieved")
                    phase_success = False
        
        except Exception as e:
            print(f"❌ Post-optimization phase failed: {e}")
            phase_success = False
        
        phase_duration = time.time() - phase_start
        self.execution_log["phases"]["post_optimization"] = {
            "success": phase_success,
            "duration": phase_duration,
            "validation_report": self.execution_log.get("validation_report")
        }
        
        print(f"\n📋 Phase 2 {'✅ COMPLETED' if phase_success else '❌ FAILED'} ({phase_duration:.3f}s)")
        return phase_success
    
    def execute_cleanup_phase(self) -> bool:
        """Execute cleanup and finalization phase."""
        print("\n🚀 PHASE 3: CLEANUP AND FINALIZATION")
        print("=" * 60)
        
        phase_start = time.time()
        phase_success = True
        
        try:
            # Step 1: Verify file count reduction
            print("\n📊 Step 1: Verifying file count reduction...")
            current_metrics = self.performance_baseline.capture_file_system_metrics()
            
            if self.execution_log.get("baseline_captured"):
                with open(self.execution_log["baseline_captured"], 'r') as f:
                    baseline_data = json.load(f)
                
                baseline_files = baseline_data["metrics"].get("total_files", 0)
                current_files = current_metrics.get("total_files", 0)
                
                if current_files < baseline_files:
                    reduction = baseline_files - current_files
                    reduction_pct = (reduction / baseline_files) * 100
                    print(f"✅ File count reduced: {baseline_files} → {current_files} (-{reduction} files, -{reduction_pct:.1f}%)")
                else:
                    print(f"⚠️  File count not reduced: {baseline_files} → {current_files}")
            else:
                print("⚠️  No baseline available for comparison")
            
            # Step 2: Verify repository size reduction
            print("\n💾 Step 2: Verifying repository size...")
            if self.execution_log.get("baseline_captured"):
                baseline_size = baseline_data["metrics"].get("repository_size_mb", 0)
                current_size = current_metrics.get("repository_size_mb", 0)
                
                if current_size > 0 and baseline_size > 0:
                    if current_size < baseline_size:
                        size_reduction = baseline_size - current_size
                        size_reduction_pct = (size_reduction / baseline_size) * 100
                        print(f"✅ Repository size reduced: {baseline_size:.1f}MB → {current_size:.1f}MB (-{size_reduction:.1f}MB, -{size_reduction_pct:.1f}%)")
                    else:
                        print(f"⚠️  Repository size not reduced: {baseline_size:.1f}MB → {current_size:.1f}MB")
                else:
                    print("⚠️  Repository size comparison not available")
            
            # Step 3: Generate final optimization report
            print("\n📄 Step 3: Generating final optimization report...")
            final_report = self._generate_final_report()
            
            with open(f"optimization_final_report_{self.execution_log['timestamp']}.json", 'w') as f:
                json.dump(final_report, f, indent=2)
            
            print(f"✅ Final report generated: optimization_final_report_{self.execution_log['timestamp']}.json")
            
        except Exception as e:
            print(f"❌ Cleanup phase failed: {e}")
            phase_success = False
        
        phase_duration = time.time() - phase_start
        self.execution_log["phases"]["cleanup"] = {
            "success": phase_success,
            "duration": phase_duration
        }
        
        print(f"\n📋 Phase 3 {'✅ COMPLETED' if phase_success else '❌ FAILED'} ({phase_duration:.3f}s)")
        return phase_success
    
    def _verify_optimization_results(self, validation_report: Dict[str, Any]) -> bool:
        """Verify that optimization achieved expected results."""
        success_criteria = {
            "imports_working": validation_report["tests"]["import_validation"]["failed"] == 0,
            "functionality_preserved": validation_report["tests"]["functionality_validation"]["failed"] == 0,
            "performance_maintained": validation_report["tests"]["performance_validation"]["performance_maintained"],
            "test_runners_working": validation_report["tests"]["test_runner_validation"]["failed"] == 0
        }
        
        print("  📋 Optimization Success Criteria:")
        for criterion, passed in success_criteria.items():
            status = "✅" if passed else "❌"
            print(f"    {status} {criterion.replace('_', ' ').title()}")
        
        return all(success_criteria.values())
    
    def _generate_final_report(self) -> Dict[str, Any]:
        """Generate comprehensive final optimization report."""
        total_duration = sum(
            phase_data.get("duration", 0) 
            for phase_data in self.execution_log["phases"].values()
        )
        
        overall_success = all(
            phase_data.get("success", False) 
            for phase_data in self.execution_log["phases"].values()
        )
        
        report = {
            "optimization_summary": {
                "timestamp": self.execution_log["timestamp"],
                "total_duration": total_duration,
                "overall_success": overall_success,
                "phases_completed": len(self.execution_log["phases"]),
                "backup_available": self.execution_log.get("backup_created") is not None
            },
            "phase_results": self.execution_log["phases"],
            "artifacts": {
                "backup_path": self.execution_log.get("backup_created"),
                "baseline_file": self.execution_log.get("baseline_captured"),
                "validation_report": self.execution_log.get("validation_report")
            },
            "recommendations": self._generate_recommendations(overall_success)
        }
        
        self.execution_log["overall_success"] = overall_success
        return report
    
    def _generate_recommendations(self, overall_success: bool) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        if overall_success:
            recommendations.extend([
                "✅ Optimization completed successfully",
                "📦 Keep backup for at least one development cycle",
                "📊 Monitor performance in production",
                "🧪 Run validation tests regularly",
                "📄 Update documentation to reflect changes"
            ])
        else:
            recommendations.extend([
                "❌ Optimization had issues - review validation report",
                "🔄 Consider rollback if critical functionality affected",
                "🔍 Investigate failed validation tests",
                "📋 Address issues before proceeding with further optimizations",
                "🛠️  Manual intervention may be required"
            ])
        
        return recommendations
    
    def run_complete_validation_cycle(self, mode: str = "full") -> bool:
        """Run the complete validation cycle."""
        print("🚀 CODEBASE OPTIMIZATION VALIDATION CYCLE")
        print("=" * 70)
        print(f"Mode: {mode.upper()}")
        print(f"Timestamp: {self.execution_log['timestamp']}")
        print("=" * 70)
        
        overall_success = True
        
        if mode in ["full", "pre"]:
            success = self.execute_pre_optimization_phase()
            overall_success = overall_success and success
            
            if not success and mode == "full":
                print("\n❌ Pre-optimization phase failed - stopping execution")
                return False
        
        if mode in ["full", "post"]:
            success = self.execute_post_optimization_phase()
            overall_success = overall_success and success
        
        if mode in ["full", "cleanup"]:
            success = self.execute_cleanup_phase()
            overall_success = overall_success and success
        
        # Final summary
        print("\n" + "=" * 70)
        print("🏁 VALIDATION CYCLE SUMMARY")
        print("=" * 70)
        
        total_duration = sum(
            phase_data.get("duration", 0) 
            for phase_data in self.execution_log["phases"].values()
        )
        
        print(f"Overall Result: {'✅ SUCCESS' if overall_success else '❌ FAILED'}")
        print(f"Total Duration: {total_duration:.3f}s")
        print(f"Phases Completed: {len(self.execution_log['phases'])}")
        
        if self.execution_log.get("backup_created"):
            print(f"Backup Available: {self.execution_log['backup_created']}")
        
        if not overall_success:
            print("\n❌ ISSUES DETECTED:")
            for phase_name, phase_data in self.execution_log["phases"].items():
                if not phase_data.get("success", True):
                    print(f"  • {phase_name.replace('_', ' ').title()} phase failed")
        
        return overall_success


def main():
    """CLI interface for validation execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Execute codebase optimization validation")
    parser.add_argument("--mode", choices=["full", "pre", "post", "cleanup"], 
                       default="full", help="Validation mode to run")
    parser.add_argument("--baseline", help="Baseline file for performance comparison")
    
    args = parser.parse_args()
    
    executor = ValidationExecutor()
    
    if args.mode == "post" and args.baseline:
        # For post-optimization with specific baseline
        success = executor.execute_post_optimization_phase(args.baseline)
    else:
        # Run specified mode
        success = executor.run_complete_validation_cycle(args.mode)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()