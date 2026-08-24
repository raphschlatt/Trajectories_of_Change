from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_release_metadata_check_accepts_current_version() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/ops/check_release_version.py",
            "v0.2.0",
        ],
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert proc.returncode == 0, proc.stderr
    assert "release metadata ok: v0.2.0" in proc.stdout
