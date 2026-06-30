# tests/test_pathcheck.py
import sys
def test_pythonpath():
    print("\n\nPYTHONPATH:", sys.path)
    assert any("src" in p for p in sys.path)
