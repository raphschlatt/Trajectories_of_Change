from __future__ import annotations

import subprocess
import sys
import tarfile
import zipfile
import os
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run(args: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        **kwargs,
    )


def test_no_local_artifacts_are_tracked() -> None:
    tracked = {
        path
        for path in _run(["git", "ls-files"]).stdout.splitlines()
        if (ROOT / path).exists()
    }

    forbidden_exact = {
        ".github/copilot-instructions.md",
        "AGENTS.md",
        ".claude/settings.local.json",
        "DATA_CONTRACT.md",
        "environment.yml",
        "GITLAB_SYNC_ROADMAP.md",
        "To Do.md",
    }
    forbidden_prefixes = (
        "archive/",
        "notebooks/",
        "plots/",
        "src/trajectories_of_change.egg-info/",
    )

    assert not (tracked & forbidden_exact)
    assert not [path for path in tracked if path.startswith(forbidden_prefixes)]


def test_local_artifact_paths_are_ignored() -> None:
    ignored = _run(
        [
            "git",
            "check-ignore",
            "AGENTS.md",
            ".claude/settings.local.json",
            "archive/research_notebooks/example.ipynb",
            "data/publications.parquet",
            "plots/example.html",
            "src/trajectories_of_change.egg-info/PKG-INFO",
        ]
    ).stdout.splitlines()

    assert "AGENTS.md" in ignored
    assert ".claude/settings.local.json" in ignored
    assert "archive/research_notebooks/example.ipynb" in ignored
    assert "data/publications.parquet" in ignored
    assert "plots/example.html" in ignored
    assert "src/trajectories_of_change.egg-info/PKG-INFO" in ignored


def test_build_artifacts_have_clean_boundaries(tmp_path: Path) -> None:
    out_dir = tmp_path / "dist"
    _run([sys.executable, "-m", "build", "--sdist", "--wheel", "--outdir", str(out_dir)])

    wheel_path = next(out_dir.glob("*.whl"))
    sdist_path = next(out_dir.glob("*.tar.gz"))

    with zipfile.ZipFile(wheel_path) as wheel:
        wheel_names = set(wheel.namelist())

    with tarfile.open(sdist_path) as sdist:
        sdist_names = {Path(member.name).as_posix() for member in sdist.getmembers()}

    forbidden_parts = {
        ".claude",
        "AGENTS.md",
            "archive",
            "plots",
            "runs",
            "notebooks",
        "__pycache__",
        "trajectories_of_change.egg-info",
        "copilot-instructions.md",
        "Anhang.tex",
        "environment.yml",
        "docs",
        "research",
        "tests",
    }

    for names in (wheel_names, sdist_names):
        for name in names:
            parts = set(Path(name).parts)
            assert parts.isdisjoint(forbidden_parts), name
            if "/data/" in name.replace("\\", "/"):
                assert "/examples/data/" in name.replace("\\", "/"), name

    assert "trajectories_of_change/__init__.py" in wheel_names
    assert "trajectories_of_change/py.typed" in wheel_names
    assert any(name.endswith(".dist-info/entry_points.txt") for name in wheel_names)
    assert any(name.endswith("/README.md") for name in sdist_names)
    assert any(name.endswith("/LICENSE") for name in sdist_names)
    assert any(name.endswith("/CITATION.cff") for name in sdist_names)
    assert any(name.endswith("/CHANGELOG.md") for name in sdist_names)
    assert any(name.endswith("/examples/quickstart_colab.ipynb") for name in sdist_names)
    assert any(name.endswith("/examples/data/publications.parquet") for name in sdist_names)
    assert any(name.endswith("/examples/data/references.parquet") for name in sdist_names)
    assert any(name.endswith("/scripts/ops/check_release_version.py") for name in sdist_names)
    assert any(name.endswith("/scripts/generate_synthetic_oracle_data.py") for name in sdist_names)

    install_target = tmp_path / "site"
    uv_executable = shutil.which("uv")
    assert uv_executable is not None
    _run(
        [
            uv_executable,
            "pip",
            "install",
            "--python",
            sys.executable,
            "--no-deps",
            "--target",
            str(install_target),
            str(wheel_path),
        ]
    )
    env = {**os.environ, "PYTHONPATH": str(install_target)}
    imported = subprocess.run(
        [
            sys.executable,
            "-c",
            "import trajectories_of_change as toc; print(toc.__version__)",
        ],
        cwd=tmp_path,
        env=env,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert imported.stdout.strip() == "0.2.0"


def test_quickstart_notebook_is_portable_json() -> None:
    import json

    notebook = ROOT / "examples" / "quickstart_colab.ipynb"
    payload = json.loads(notebook.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in payload.get("cells", [])
    )

    assert payload["nbformat"] >= 4
    assert "sys.path" not in source
    assert "trajectories_of_change.metrics_" not in source
    assert "trajectories_of_change.multimetric" not in source
    assert "trajectories_of_change.contract" not in source
    assert "trajectories_of_change.data_loader" not in source
    assert "DEFAULT_TOP_K_KLD_TERMS" in source
    assert "top_k_kld_terms=20" not in source


def test_project_docs_use_uv_not_conda() -> None:
    checked_paths = [
        ROOT / "README.md",
        ROOT / "docs",
        ROOT / "examples" / "README.md",
        ROOT / "scripts",
        ROOT / ".github" / "workflows",
    ]
    offenders: list[str] = []
    text_suffixes = {".md", ".py", ".yml", ".yaml", ".toml", ".json", ".cff", ".txt"}
    for path in checked_paths:
        files = path.rglob("*") if path.is_dir() else [path]
        for file_path in files:
            if not file_path.is_file():
                continue
            if "__pycache__" in file_path.parts or file_path.suffix not in text_suffixes:
                continue
            text = file_path.read_text(encoding="utf-8")
            lowered = text.lower()
            if "conda" in lowered or "environment.yml" in lowered:
                offenders.append(str(file_path.relative_to(ROOT)))

    assert offenders == []
