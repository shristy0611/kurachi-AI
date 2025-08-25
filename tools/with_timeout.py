#!/usr/bin/env python3
"""
Cross-platform timeout wrapper for test execution
Usage: python tools/with_timeout.py 30 python tests/smoke/smoke_test.py
"""
import subprocess
import sys
import signal
import os
from typing import List


def run_with_timeout(timeout_seconds: int, command: List[str]) -> int:
    """Run command with timeout, return exit code"""
    try:
        # Use subprocess timeout (Python 3.3+)
        result = subprocess.run(
            command,
            timeout=timeout_seconds,
            capture_output=False,  # Let output go to terminal
            text=True
        )
        return result.returncode
        
    except subprocess.TimeoutExpired:
        print(f"\n⏰ Command timed out after {timeout_seconds} seconds")
        return 124  # Standard timeout exit code
        
    except KeyboardInterrupt:
        print(f"\n🛑 Command interrupted by user")
        return 130  # Standard interrupt exit code
        
    except Exception as e:
        print(f"\n💥 Error running command: {e}")
        return 1


def main():
    """Main entry point"""
    if len(sys.argv) < 3:
        print("Usage: python tools/with_timeout.py <timeout_seconds> <command> [args...]")
        print("Example: python tools/with_timeout.py 30 python tests/smoke/smoke_test.py")
        sys.exit(1)
    
    try:
        timeout_seconds = int(sys.argv[1])
        command = sys.argv[2:]
        
        if timeout_seconds <= 0:
            print("Error: Timeout must be positive")
            sys.exit(1)
            
        print(f"🕐 Running command with {timeout_seconds}s timeout: {' '.join(command)}")
        exit_code = run_with_timeout(timeout_seconds, command)
        sys.exit(exit_code)
        
    except ValueError:
        print("Error: First argument must be a number (timeout in seconds)")
        sys.exit(1)


if __name__ == "__main__":
    main()