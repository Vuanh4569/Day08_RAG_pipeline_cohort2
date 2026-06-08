import sys
from pathlib import Path

# Add current folder to path to import import_helper
_parent_dir = str(Path(__file__).resolve().parent)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from import_helper import load_personal_module

# Load personal task module
_module = load_personal_module("task10_generation")

# Expose everything to the root namespace
globals().update({k: v for k, v in _module.__dict__.items() if not k.startswith('_')})

if __name__ == "__main__":
    q = "Chi Dân bị bắt vì sử dụng ma tuý như thế nào?"
    res = generate_with_citation(q)
    print(f"\nQuery: {q}")
    print(f"Answer:\n{res['answer']}")
    print(f"Retrieval source: {res['retrieval_source']}")
