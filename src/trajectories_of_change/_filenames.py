"""Internal helpers for deterministic cross-platform artifact names."""

from __future__ import annotations

from hashlib import sha256
import re


_WINDOWS_RESERVED = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{number}" for number in range(1, 10)),
    *(f"LPT{number}" for number in range(1, 10)),
}


def safe_filename_component(value: object, *, max_length: int = 80, fallback: str = "target") -> str:
    """Return a stable ASCII filename component that is safe on Windows."""
    original = str(value).strip()
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", original)
    slug = re.sub(r"_+", "_", slug).strip(" ._")
    if not slug:
        slug = fallback
    if slug.split(".", 1)[0].upper() in _WINDOWS_RESERVED:
        slug = f"_{slug}"
    if len(slug) > max_length:
        digest = sha256(original.encode("utf-8")).hexdigest()[:10]
        slug = f"{slug[: max_length - 11].rstrip(' ._')}_{digest}"
    return slug or fallback
