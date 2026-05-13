"""Install only missing or version-mismatched packages from requirements.txt

This script reads `backend/requirements.txt`, checks installed package
versions in the current Python environment, and installs packages that are
missing or whose versions differ from the pinned version.

Run with the workspace venv Python:
  g:/VisionRAG/.venv/Scripts/python.exe backend/scripts/install_missing.py
"""
from __future__ import annotations
import subprocess
import sys
from pathlib import Path


def parse_requirements(path: Path) -> dict[str, str]:
    reqs: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "==" in line:
            name, ver = line.split("==", 1)
            reqs[name.strip()] = ver.strip()
        else:
            # skip loose requirements
            parts = line.split()
            reqs[parts[0]] = ""
    return reqs


def get_installed_version(pkg: str) -> str | None:
    try:
        r = subprocess.run([sys.executable, "-m", "pip", "show", pkg], capture_output=True, text=True)
        if r.returncode != 0 or not r.stdout:
            return None
        for line in r.stdout.splitlines():
            if line.startswith("Version:"):
                return line.split("Version:", 1)[1].strip()
    except Exception:
        return None


def install_package(spec: str) -> int:
    print(f"Installing: {spec}")
    return subprocess.run([sys.executable, "-m", "pip", "install", spec], check=False).returncode


def main():
    # repo root: two levels above this script ( .. / .. )
    root = Path(__file__).resolve().parents[2]
    req_path = root / "backend" / "requirements.txt"
    if not req_path.exists():
        print(f"requirements file not found: {req_path}")
        sys.exit(1)

    reqs = parse_requirements(req_path)
    to_install: list[str] = []

    for name, pinned in reqs.items():
        inst = get_installed_version(name)
        if inst is None:
            spec = f"{name}=={pinned}" if pinned else name
            to_install.append(spec)
            print(f"Missing: {name} (will install {spec})")
        elif pinned and inst != pinned:
            print(f"Version mismatch for {name}: installed {inst}, required {pinned} (will upgrade)")
            to_install.append(f"{name}=={pinned}")
        else:
            print(f"OK: {name}=={inst}")

    if not to_install:
        print("All requirements satisfied.")
        return

    for spec in to_install:
        rc = install_package(spec)
        if rc != 0:
            print(f"Failed to install {spec} (exit {rc})")


if __name__ == "__main__":
    main()
