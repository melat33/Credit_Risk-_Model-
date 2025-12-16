#!/usr/bin/env python
"""
Simple test runner for Bati Bank Credit Risk Model
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

if __name__ == "__main__":
    print("Running Bati Bank Credit Risk Model Tests...")
    print("=" * 60)

    # Run the test module directly
    import test_model_pipeline

    # This will run the tests when the module is imported
    # since the module has __name__ == "__main__" block
