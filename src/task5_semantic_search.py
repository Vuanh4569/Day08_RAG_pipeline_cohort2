import sys
from pathlib import Path

# Add current folder to path to import import_helper
_parent_dir = str(Path(__file__).resolve().parent)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from import_helper import load_personal_module

# Load personal task module
_module = load_personal_module("task5_semantic_search")

# Expose everything to the root namespace
globals().update({k: v for k, v in _module.__dict__.items() if not k.startswith('_')})

if __name__ == "__main__":
    test_queries = [
        "hình phạt cho tội tàng trữ ma tuý",
        "nghệ sĩ Chi Dân bị bắt"
    ]
    for q in test_queries:
        print(f"\nQuery: {q}")
        print("-" * 50)
        res = semantic_search(q, top_k=3)
        for r in res:
            print(f"[{r['score']:.4f}] {r['content'][:150]}...")
