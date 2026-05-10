import subprocess
import sys
from pathlib import Path


def test_full_script_passes():
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "tests.py"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=300,
    )
    output = "\n".join(part for part in (result.stdout, result.stderr) if part)
    assert result.returncode == 0, output
