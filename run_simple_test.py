"""
Simple test runner to debug the import issue
"""

import sys
import os
from pathlib import Path

print("="*60)
print("DEBUGGING IMPORT ISSUE")
print("="*60)

# Show current directory
print(f"Current directory: {os.getcwd()}")
print(f"Python executable: {sys.executable}")

# Check if src exists
project_root = Path(os.getcwd())
src_path = project_root / "src"
print(f"\nChecking for src directory at: {src_path}")

if src_path.exists():
    print("✅ src directory exists")
    print("Files in src/:")
    for item in src_path.iterdir():
        print(f"  - {item.name}")
    
    # Check features
    features_path = src_path / "features"
    if features_path.exists():
        print("\nFiles in src/features/:")
        for item in features_path.iterdir():
            print(f"  - {item.name}")
    else:
        print("\n❌ src/features/ does not exist")
else:
    print("❌ src directory does not exist")

# Add project root to Python path
print(f"\nAdding to Python path: {project_root}")
sys.path.insert(0, str(project_root))

print(f"\nPython path:")
for i, path in enumerate(sys.path[:5]):
    print(f"  {i}: {path}")

# Try to import
print("\nTrying to import...")
try:
    # Try absolute import
    from src.features import AggregationEngine
    print("✅ SUCCESS: Imported AggregationEngine")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    
    # Try relative import
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "aggregations", 
            str(project_root / "src" / "features" / "aggregations.py")
        )
        if spec:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            AggregationEngine = module.AggregationEngine
            print("✅ SUCCESS: Loaded module directly")
        else:
            print("❌ Could not load module")
    except Exception as e2:
        print(f"❌ Direct load failed: {e2}")

print("\n" + "="*60)
print("DEBUG COMPLETE")
print("="*60)