#!/usr/bin/env python3
"""
CI Validation Gate - Ensures validation system passes before allowing builds
Run this in CI to fail builds if validation or core tests fail
"""
import sys
import subprocess
import time
from pathlib import Path

def run_command(cmd, description, timeout=120):
    """Run a command with timeout and proper error handling"""
    print(f"🔍 {description}...")
    print(f"   Command: {' '.join(cmd)}")
    
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=Path(__file__).parent.parent
        )
        duration = time.time() - start_time
        
        if result.returncode == 0:
            print(f"   ✅ {description} PASSED ({duration:.2f}s)")
            return True
        else:
            print(f"   ❌ {description} FAILED ({duration:.2f}s)")
            print(f"   STDOUT: {result.stdout[-500:]}")  # Last 500 chars
            print(f"   STDERR: {result.stderr[-500:]}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"   ⏰ {description} TIMED OUT after {timeout}s")
        return False
    except Exception as e:
        print(f"   💥 {description} ERROR: {e}")
        return False

def main():
    """Run CI validation gate"""
    print("🚀 CI VALIDATION GATE")
    print("=" * 50)
    
    # Core validation checks
    checks = [
        {
            "cmd": [sys.executable, "tools/validation_system.py"],
            "description": "Core Validation System",
            "timeout": 60
        },
        {
            "cmd": [sys.executable, "test_runner_fast.py"],
            "description": "Fast Test Runner",
            "timeout": 30
        },
        {
            "cmd": [sys.executable, "tools/test_validation_system.py"],
            "description": "Validation System Self-Test",
            "timeout": 30
        }
    ]
    
    # Optional checks (don't fail CI but report status)
    optional_checks = [
        {
            "cmd": [sys.executable, "-c", "import spacy; print('spaCy available')"],
            "description": "spaCy Models (Optional)",
            "timeout": 10
        }
    ]
    
    passed = 0
    failed = 0
    
    # Run required checks
    print("\n📋 REQUIRED CHECKS")
    print("-" * 30)
    for check in checks:
        if run_command(check["cmd"], check["description"], check.get("timeout", 60)):
            passed += 1
        else:
            failed += 1
    
    # Run optional checks
    print("\n📋 OPTIONAL CHECKS")
    print("-" * 30)
    for check in optional_checks:
        run_command(check["cmd"], check["description"], check.get("timeout", 10))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 CI VALIDATION SUMMARY")
    print("=" * 50)
    print(f"Required Checks: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("✅ CI VALIDATION GATE PASSED")
        print("🚀 Build can proceed")
        return 0
    else:
        print("❌ CI VALIDATION GATE FAILED")
        print("🛑 Build should be blocked")
        return 1

if __name__ == "__main__":
    sys.exit(main())