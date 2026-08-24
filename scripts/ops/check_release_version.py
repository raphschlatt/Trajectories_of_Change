from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]


def fail(message: str) -> None:
    print(f"release metadata check failed: {message}", file=sys.stderr)
    raise SystemExit(1)


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        fail(f"missing {path.relative_to(ROOT)}")


def normalize_tag(raw_tag: str) -> str:
    tag = raw_tag.strip()
    if tag.startswith("refs/tags/"):
        tag = tag.removeprefix("refs/tags/")
    if not re.fullmatch(r"v\d+\.\d+\.\d+", tag):
        fail(f"tag must look like vX.Y.Z, got {raw_tag!r}")
    return tag.removeprefix("v")


def pyproject_version() -> str:
    data = tomllib.loads(read_text(ROOT / "pyproject.toml"))
    try:
        return str(data["project"]["version"])
    except KeyError:
        fail("pyproject.toml missing [project].version")


def package_version() -> str:
    init_text = read_text(ROOT / "src" / "trajectories_of_change" / "__init__.py")
    match = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', init_text, re.MULTILINE)
    if not match:
        fail("src/trajectories_of_change/__init__.py missing __version__")
    return match.group(1)


def changelog_release(version: str) -> tuple[str, str]:
    changelog = read_text(ROOT / "CHANGELOG.md")
    pattern = re.compile(
        rf"^## \[{re.escape(version)}\] - (?P<date>\d{{4}}-\d{{2}}-\d{{2}})\s*$",
        re.MULTILINE,
    )
    match = pattern.search(changelog)
    if not match:
        fail(f"CHANGELOG.md missing release section for {version}")
    body_start = match.end()
    next_match = re.search(r"^## \[", changelog[body_start:], re.MULTILINE)
    body_end = body_start + next_match.start() if next_match else len(changelog)
    body = changelog[body_start:body_end].strip()
    if not body:
        fail(f"CHANGELOG.md release section for {version} is empty")
    return match.group("date"), body


def yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(read_text(path)) or {}
    except yaml.YAMLError as exc:
        fail(f"{path.name} is invalid YAML: {exc}")
    if not isinstance(payload, dict):
        fail(f"{path.name} must contain a YAML mapping")
    return payload


def json_mapping(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(read_text(path))
    except json.JSONDecodeError as exc:
        fail(f"{path.name} is invalid JSON: {exc}")
    if not isinstance(payload, dict):
        fail(f"{path.name} must contain a JSON object")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("tag", help="Release tag such as v0.1.0")
    parser.add_argument("--notes-out", type=Path, help="Write release notes from CHANGELOG.md")
    args = parser.parse_args()

    version = normalize_tag(args.tag)
    release_date, release_notes = changelog_release(version)

    if pyproject_version() != version:
        fail(f"pyproject.toml version does not match v{version}")
    if package_version() != version:
        fail(f"package __version__ does not match v{version}")

    citation = yaml_mapping(ROOT / "CITATION.cff")
    if str(citation.get("version", "")).strip() != version:
        fail("CITATION.cff version does not match release tag")
    if str(citation.get("date-released", "")).strip() != release_date:
        fail("CITATION.cff date-released does not match CHANGELOG.md")

    zenodo = json_mapping(ROOT / ".zenodo.json")
    if str(zenodo.get("version", "")).strip() != version:
        fail(".zenodo.json version does not match release tag")
    if str(zenodo.get("publication_date", "")).strip() != release_date:
        fail(".zenodo.json publication_date does not match CHANGELOG.md")
    for key in ("title", "description", "creators", "upload_type", "access_right", "language"):
        if not zenodo.get(key):
            fail(f".zenodo.json missing {key!r}")

    if args.notes_out:
        args.notes_out.parent.mkdir(parents=True, exist_ok=True)
        args.notes_out.write_text(release_notes + "\n", encoding="utf-8")

    print(f"release metadata ok: v{version} ({release_date})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
