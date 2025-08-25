#!/usr/bin/env python3
"""
Test script to verify validation system functionality.
This script tests the validation system components to ensure they work correctly.
"""

import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.validation_system import ValidationSystem
from tools.backup_system import BackupSystem

def test_backup_system():
    """Test backup system functionality."""
    print("🧪 Testing Backup System...")
    
    backup_system = BackupSystem()
    
    # Test backup creation
    try:
        backup_path = backup_system.create_backup()
        print(f"  ✅ Backup creation successful: {backup_path}")
        
        # Test backup listing
        backups = backup_system.list_backups()
        if backups:
            print(f"  ✅ Backup listing successful: {len(backups)} backups found")
        else:
            print("  ⚠️  No backups found in listing")
        
        return True
    except Exception as e:
        print(f"  ❌ Backup system test failed: {e}")
        return False

def test_validation_system():
    """Test validation system functionality."""
    print("🧪 Testing Validation System...")
    
    validator = ValidationSystem()
    
    # Test import validation
    try:
        import_results = validator.validate_imports()
        print(f"  ✅ Import validation completed: {import_results['passed']} passed, {import_results['failed']} failed")
    except Exception as e:
        print(f"  ❌ Import validation failed: {e}")
        return False
    
    # Test functionality validation
    try:
        func_results = validator.validate_functionality()
        print(f"  ✅ Functionality validation completed: {func_results['passed']} passed, {func_results['failed']} failed")
    except Exception as e:
        print(f"  ❌ Functionality validation failed: {e}")
        return False
    
    # Test test runner validation
    try:
        runner_results = validator.validate_test_runners()
        print(f"  ✅ Test runner validation completed: {runner_results['passed']} passed, {runner_results['failed']} failed")
    except Exception as e:
        print(f"  ❌ Test runner validation failed: {e}")
        return False
    
    return True

def test_performance_validation():
    """Test performance validation functionality."""
    print("🧪 Testing Performance Validation...")
    
    validator = ValidationSystem()
    
    try:
        # Test performance validation without baseline
        perf_results = validator.validate_performance()
        print(f"  ✅ Performance validation completed: {perf_results['performance_maintained']}")
        return True
    except Exception as e:
        print(f"  ❌ Performance validation failed: {e}")
        return False

def test_validation_report_generation():
    """Test validation report generation."""
    print("🧪 Testing Validation Report Generation...")
    
    validator = ValidationSystem()
    
    try:
        # Run a quick validation to populate report data
        validator.validate_imports()
        
        # Test report saving
        report_file = validator.save_report("test_validation_report.json")
        
        if Path(report_file).exists():
            print(f"  ✅ Report generation successful: {report_file}")
            # Clean up test file
            Path(report_file).unlink()
            return True
        else:
            print("  ❌ Report file not created")
            return False
    except Exception as e:
        print(f"  ❌ Report generation failed: {e}")
        return False

def main():
    """Run all validation system tests."""
    print("🚀 VALIDATION SYSTEM TEST SUITE")
    print("=" * 50)
    
    tests = [
        ("Backup System", test_backup_system),
        ("Validation System", test_validation_system),
        ("Performance Validation", test_performance_validation),
        ("Report Generation", test_validation_report_generation)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n🔍 {test_name}")
        print("-" * 30)
        
        start_time = time.time()
        try:
            success = test_func()
            duration = time.time() - start_time
            
            if success:
                print(f"  ✅ {test_name} PASSED ({duration:.3f}s)")
                passed += 1
            else:
                print(f"  ❌ {test_name} FAILED ({duration:.3f}s)")
                failed += 1
        except Exception as e:
            duration = time.time() - start_time
            print(f"  ❌ {test_name} ERROR ({duration:.3f}s): {e}")
            failed += 1
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    print(f"Total Tests: {passed + failed}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success Rate: {(passed / (passed + failed)) * 100:.1f}%")
    
    if failed == 0:
        print("✅ All validation system tests passed!")
        return True
    else:
        print(f"❌ {failed} validation system tests failed!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)