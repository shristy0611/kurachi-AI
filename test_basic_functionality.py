#!/usr/bin/env python3
"""
Basic functionality test for Kurachi AI
Tests core components without complex dependencies
"""

import sys
import os
sys.path.append('.')

def test_config():
    """Test configuration loading"""
    try:
        from config import config
        print("✅ Configuration loaded successfully")
        print(f"   App: {config.app}")
        print(f"   AI: {config.ai}")
        print(f"   Database: {config.database}")
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_logger():
    """Test logging system"""
    try:
        from utils.logger import get_logger
        logger = get_logger('test')
        logger.info('Test log message')
        print("✅ Logger working correctly")
        return True
    except Exception as e:
        print(f"❌ Logger test failed: {e}")
        return False

def test_database():
    """Test database connection"""
    try:
        from models.database import db_manager
        print("✅ Database manager initialized")
        print(f"   Database path: {db_manager.db_path}")
        return True
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False

def test_utils():
    """Test utility modules"""
    try:
        from utils import helpers
        from utils import exceptions
        from utils import enum_coercion
        print("✅ Utility modules imported successfully")
        return True
    except Exception as e:
        print(f"❌ Utils test failed: {e}")
        return False

def test_streamlit():
    """Test Streamlit functionality"""
    try:
        import streamlit as st
        print(f"✅ Streamlit imported successfully (v{st.__version__})")
        return True
    except Exception as e:
        print(f"❌ Streamlit test failed: {e}")
        return False

def test_frontend_access():
    """Test if frontend is accessible"""
    try:
        import requests
        response = requests.get('http://localhost:8501', timeout=5)
        if response.status_code == 200:
            print("✅ Frontend is accessible at http://localhost:8501")
            return True
        else:
            print(f"❌ Frontend returned status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Frontend test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧠 Kurachi AI - Basic Functionality Test")
    print("=" * 50)
    
    tests = [
        ("Configuration", test_config),
        ("Logger", test_logger),
        ("Database", test_database),
        ("Utilities", test_utils),
        ("Streamlit", test_streamlit),
        ("Frontend Access", test_frontend_access),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Testing {test_name}...")
        if test_func():
            passed += 1
        else:
            print(f"   ⚠️  {test_name} test failed")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Kurachi AI is ready to use.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
