"""
tests/conftest.py

Shared pytest fixtures and configuration.
"""
import sys
from pathlib import Path

# Ensure project root is on sys.path for all tests
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
