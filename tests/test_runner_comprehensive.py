#!/usr/bin/env python3
"""
Comprehensive Test Runner for Kurachi AI
Runs the complete test suite including unit, integration, and smoke tests
Target: Complete validation with full coverage
"""

import time
import sys
import os
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

class ComprehensiveTestRunner:
    """Comprehensive test runner with full coverage"""
    
    def __init__(self):
        self.results = []
        self.start_time = time.time()
        
    def run_test_suite(self, name: str, command: List[str], timeout: int = 300) -> Dict[str, Any]:
        """Run a test suite with timeout and capture results"""
        print(f"🧪 Running {name}...")
        start_time = time.time()
        
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=Path(__file__).parent.parent
            )
            
            duration = time.time() - start_time
            success = result.returncode == 0
            
            test_result = {
                'name': name,
                'duration': duration,
                'success': success,
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
            
            self.results.append(test_result)
            
            if success:
                print(f"✅ {name} passed ({duration:.2f}s)")
            else:
                print(f"❌ {name} failed ({duration:.2f}s)")
                if result.stderr:
                    print(f"   Error: {result.stderr[:200]}...")
            
            return test_result
            
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            print(f"⏰ {name} timed out after {timeout}s")
            test_result = {
                'name': name,
                'duration': duration,
                'success': False,
                'returncode': -1,
                'stdout': '',
                'stderr': f'Test timed out after {timeout}s'
            }
            self.results.append(test_result)
            return test_result
            
        except Exception as e:
            duration = time.time() - start_time
            print(f"💥 {name} crashed: {str(e)}")
            test_result = {
                'name': name,
                'duration': duration,
                'success': False,
                'returncode': -2,
                'stdout': '',
                'stderr': str(e)
            }
            self.results.append(test_result)
            return test_result
    
    def generate_report(self) -> str:
        """Generate comprehensive test report"""
        total_time = time.time() - self.start_time
        passed = len([r for r in self.results if r['success']])
        failed = len([r for r in self.results if not r['success']])
        
        report = f"""
🚀 COMPREHENSIVE TEST SUITE REPORT
{'='*60}
📊 Summary:
  • Total Test Suites: {len(self.results)}
  • Passed: {passed}
  • Failed: {failed}
  • Total Time: {total_time:.2f}s
  • Success Rate: {(passed/len(self.results)*100) if self.results else 0:.1f}%

📈 Test Suite Results:
"""
        
        for result in self.results:
            status_emoji = "✅" if result['success'] else "❌"
            report += f"  {status_emoji} {result['name']}: {result['duration']:.2f}s"
            if not result['success']:
                report += f" (Exit code: {result['returncode']})"
            report += "\n"
        
        if failed > 0:
            report += "\n❌ Failed Test Details:\n"
            for result in [r for r in self.results if not r['success']]:
                report += f"  • {result['name']}:\n"
                if result['stderr']:
                    report += f"    Error: {result['stderr'][:300]}...\n"
        
        return report

def main():
    """Run comprehensive test suite"""
    print("🚀 Kurachi AI - Comprehensive Test Suite")
    print("Running complete validation with full coverage")
    print("=" * 60)
    
    runner = ComprehensiveTestRunner()
    
    # Define comprehensive test suites
    test_suites = [
        {
            'name': 'Fast Core Tests',
            'command': [sys.executable, 'tests/test_runner_fast.py'],
            'timeout': 30
        },
        {
            'name': 'Unit Tests',
            'command': [sys.executable, '-m', 'pytest', 'tests/unit/', '-v', '--tb=short'],
            'timeout': 120
        },
        {
            'name': 'Integration Tests',
            'command': [sys.executable, '-m', 'pytest', 'tests/integration/', '-v', '--tb=short'],
            'timeout': 180
        },
        {
            'name': 'Smoke Tests',
            'command': [sys.executable, '-m', 'pytest', 'tests/smoke/', '-v', '--tb=short'],
            'timeout': 300
        },
        {
            'name': 'Setup Verification',
            'command': [sys.executable, 'tests/test_setup.py'],
            'timeout': 60
        }
    ]
    
    # Run all test suites
    print(f"Running {len(test_suites)} test suites...\n")
    
    for suite in test_suites:
        runner.run_test_suite(
            suite['name'],
            suite['command'],
            suite.get('timeout', 300)
        )
        print()  # Add spacing between test suites
    
    # Generate and display report
    print(runner.generate_report())
    
    # Calculate final results
    total_time = time.time() - runner.start_time
    passed = len([r for r in runner.results if r['success']])
    total = len(runner.results)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"comprehensive_test_results_{timestamp}.json"
    
    import json
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'summary': {
                'total_suites': total,
                'passed_suites': passed,
                'failed_suites': total - passed,
                'total_time': total_time,
                'success_rate': (passed/total*100) if total > 0 else 0
            },
            'results': runner.results
        }, f, indent=2)
    
    print(f"\n📄 Results saved to: {results_file}")
    
    # Final summary
    if passed == total:
        print(f"\n🎉 SUCCESS: All {total} test suites passed in {total_time:.2f}s!")
        print("✅ System is ready for production deployment")
    else:
        failed = total - passed
        print(f"\n⚠️  ISSUES: {failed} test suite(s) failed out of {total}")
        print("❌ System needs attention before deployment")
    
    # Performance feedback
    if total_time < 300:  # 5 minutes
        print("⚡ Excellent test performance!")
    elif total_time < 600:  # 10 minutes
        print("👍 Good test performance")
    else:
        print("🐌 Test suite is slow - consider optimization")
    
    sys.exit(0 if passed == total else 1)

if __name__ == "__main__":
    main()