"""Smoke tests: verify project structure is intact."""
from pathlib import Path

ROOT = Path(__file__).parent.parent


def test_src_vision_exists():
    assert (ROOT / "src" / "vision").is_dir()


def test_src_rag_exists():
    assert (ROOT / "src" / "rag").is_dir()


def test_data_raw_exists():
    assert (ROOT / "data" / "raw").is_dir()


def test_ci_workflow_exists():
    assert (ROOT / ".github" / "workflows" / "ci.yml").is_file()
