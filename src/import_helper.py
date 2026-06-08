"""
Helper to dynamically import tasks from personal submission directories,
allowing the root-level scripts to delegate execution to the active student's code.
"""

import sys
import importlib.util
import types
from pathlib import Path

def load_personal_module(task_name: str):
    """
    Finds and imports a module from the personal submissions folder dynamically.
    Prioritizes Vu Anh's folder locally, and falls back to other members if not found.
    """
    personal_root = Path(__file__).resolve().parent.parent / "personal_submission"
    
    # Prioritized folder candidate list
    candidates = [
        "2A202600571 - Hà Vũ Anh",
        "2A202600802 - Phạm Đình Phúc",
        "2A202600758-NguyenTuanAnh-Day08",
        "cung"
    ]
    
    for c in candidates:
        candidate_path = personal_root / c
        src_dir = candidate_path / "src"
        file_path = src_dir / f"{task_name}.py"
        
        if file_path.exists():
            # Add personal src to sys.path so relative imports inside the module work
            if str(src_dir) not in sys.path:
                sys.path.insert(0, str(src_dir))
                
            # Create a virtual package 'personal_active' to support relative imports
            pkg_name = "personal_active"
            if pkg_name not in sys.modules:
                pkg_module = types.ModuleType(pkg_name)
                pkg_module.__path__ = [str(src_dir)]
                pkg_module.__package__ = pkg_name
                sys.modules[pkg_name] = pkg_module

            # Load the module under the package namespace
            full_module_name = f"{pkg_name}.{task_name}"
            
            # If the module was already loaded, return it
            if full_module_name in sys.modules:
                module = sys.modules[full_module_name]
                # Ensure aliases are also in sys.modules
                sys.modules[task_name] = module
                sys.modules[f"src.{task_name}"] = module
                return module
                
            spec = importlib.util.spec_from_file_location(full_module_name, str(file_path))
            if spec is not None and spec.loader is not None:
                module = importlib.util.module_from_spec(spec)
                module.__package__ = pkg_name
                
                # Register in sys.modules under all expected names
                sys.modules[full_module_name] = module
                sys.modules[task_name] = module
                sys.modules[f"src.{task_name}"] = module
                
                spec.loader.exec_module(module)
                return module
                
    raise ModuleNotFoundError(f"Could not find personal submission for {task_name}")
