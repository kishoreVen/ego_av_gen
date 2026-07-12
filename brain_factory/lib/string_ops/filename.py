from __future__ import annotations


def sanitise_filename(name: str) -> str:
    """Turn an action name into a safe directory-name fragment."""
    return "".join(
        c if c.isalnum() or c in ("-", "_") else "_" for c in name.lower()
    ).strip("_")
