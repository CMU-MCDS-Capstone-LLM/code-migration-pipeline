from pathlib import Path


def check_path(p: Path) -> Path:
    p.resolve()
    assert p.exists(), f"Path doesn't exist: {p}"
    return p
