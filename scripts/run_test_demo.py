"""
Example test run showcasing the test suite

This file demonstrates how to run tests and what to expect.
"""

import subprocess
import sys


def run_command(cmd: list, description: str) -> None:
    """Run a command and display results"""
    print(f"\n{'='*70}")
    print(f"  {description}")
    print(f"{'='*70}\n")
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"\n⚠️  Some tests may need AWS credentials or dependencies")
    print()


def main():
    """Run example test scenarios"""
    
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║          AWS ML Training - Test Suite Demo                      ║
║                                                                  ║
║  This demonstrates our production-ready testing approach        ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    examples = [
        {
            "cmd": ["poetry", "run", "pytest", "tests/unit/test_preprocessing.py", "-v", "--tb=short"],
            "description": "Example 1: Unit Tests - Data Preprocessing"
        },
        {
            "cmd": ["poetry", "run", "pytest", "tests/unit/test_evaluation.py", "-v", "--tb=short", "-k", "test_calculate_binary_metrics"],
            "description": "Example 2: Unit Tests - Model Evaluation Metrics"
        },
        {
            "cmd": ["poetry", "run", "pytest", "tests/unit", "-v", "--cov=src/ml_toolkit", "--cov-report=term-missing", "--tb=short"],
            "description": "Example 3: Full Unit Test Suite with Coverage"
        },
        {
            "cmd": ["poetry", "run", "pytest", "tests/", "-v", "-m", "not integration", "--maxfail=3"],
            "description": "Example 4: All Non-Integration Tests (Fast)"
        },
    ]
    
    print("\n📋 Available Test Categories:\n")
    print("  • Unit Tests       - Fast, no external dependencies")
    print("  • Integration Tests - E2E with AWS SageMaker (requires credentials)")
    print("  • Slow Tests       - Longer-running validation tests")
    print("\n")
    
    for i, example in enumerate(examples, 1):
        run_command(example["cmd"], example["description"])
        
        if i < len(examples):
            response = input("  Press Enter to continue (or 'q' to quit)... ")
            if response.lower() == 'q':
                break
    
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║  ✅ Test Demo Complete!                                          ║
║                                                                  ║
║  For more test options:                                          ║
║    • make test          - Run all tests                          ║
║    • make test-unit     - Fast unit tests only                   ║
║    • make coverage      - Generate HTML coverage report          ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
    """)


if __name__ == '__main__':
    main()
