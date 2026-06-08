import sys
import asyncio
import inspect
from pathlib import Path

# Add current folder to path to import import_helper
_parent_dir = str(Path(__file__).resolve().parent)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from import_helper import load_personal_module

# Load personal task module
_module = load_personal_module("task2_crawl_news")

# Expose everything to the root namespace
globals().update({k: v for k, v in _module.__dict__.items() if not k.startswith('_')})

if __name__ == "__main__":
    if hasattr(_module, "crawl_all"):
        if inspect.iscoroutinefunction(_module.crawl_all):
            asyncio.run(_module.crawl_all())
        else:
            _module.crawl_all()
