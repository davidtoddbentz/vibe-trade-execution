"""Guard test: compiler must never use model_dump().

The compiler pipeline should use typed Pydantic models end-to-end.
Any use of model_dump() indicates a regression to dict-based processing.
"""

import ast
from pathlib import Path


COMPILER_DIR = Path(__file__).resolve().parent.parent.parent / "src" / "translator" / "compiler"


def _collect_python_files(directory: Path) -> list[Path]:
    """Collect all .py files in directory (non-recursive)."""
    return sorted(directory.glob("*.py"))


def test_no_model_dump_in_compiler():
    """Ensure no .model_dump() calls exist in the compiler package."""
    violations = []
    for py_file in _collect_python_files(COMPILER_DIR):
        source = py_file.read_text()
        tree = ast.parse(source, filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Attribute) and func.attr == "model_dump":
                    violations.append(
                        f"{py_file.name}:{node.lineno}: .model_dump() call"
                    )
    assert not violations, (
        "compiler must not use model_dump() — found:\n"
        + "\n".join(f"  {v}" for v in violations)
    )


def test_no_model_dump_in_translator():
    """Ensure no .model_dump() calls in translator.py (the orchestrator)."""
    translator_file = COMPILER_DIR.parent / "translator.py"
    source = translator_file.read_text()
    tree = ast.parse(source, filename=str(translator_file))
    violations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "model_dump":
                violations.append(f"translator.py:{node.lineno}: .model_dump() call")
    assert not violations, (
        "translator must not use model_dump() — found:\n"
        + "\n".join(f"  {v}" for v in violations)
    )


def test_dead_code_deleted():
    """Ensure dead visitors and pipeline were deleted."""
    translator_dir = COMPILER_DIR.parent
    visitors_dir = translator_dir / "visitors"

    dead_files = [
        translator_dir / "pipeline.py",
        translator_dir / "context.py",
        visitors_dir / "indicator_collector.py",
        visitors_dir / "state_extractor.py",
    ]

    surviving = [f for f in dead_files if f.exists()]
    assert not surviving, (
        "Dead code files still exist:\n"
        + "\n".join(f"  {f.name}" for f in surviving)
    )
