#!/usr/bin/env python3
import re
import sys
from pathlib import Path

# Paths to files to update
PYPROJECT = Path("pyproject.toml")
CONF_PY = Path("docs/source/conf.py")
INIT_PY = Path("yarp/__init__.py")


def update_pyproject(version):
    content = PYPROJECT.read_text()
    new_content = re.sub(r'version\s*=\s*"[^"]+"', f'version = "{version}"', content)
    PYPROJECT.write_text(new_content)


def update_conf_py(version):
    content = CONF_PY.read_text()
    # Update 'release' and/or 'version' variables
    new_content = re.sub(r'release\s*=\s*"[^"]+"', f'release = "{version}"', content)
    new_content = re.sub(
        r'version\s*=\s*"[^"]+"', f'version = "{version}"', new_content
    )
    CONF_PY.write_text(new_content)


def update_init_py(version):
    content = INIT_PY.read_text()
    new_content = re.sub(
        r'__version__\s*=\s*"[^"]+"', f'__version__ = "{version}"', content
    )
    INIT_PY.write_text(new_content)


def main():
    version = sys.argv[1] if len(sys.argv) > 1 else None
    if not version:
        print("Usage: python update_version.py <new_version>")
        sys.exit(1)
    print(f"Updating version to: {version}")
    update_pyproject(version)
    update_conf_py(version)
    update_init_py(version)
    print(
        "Version updated in pyproject.toml, docs/source/conf.py, and yarp/__init__.py"
    )


if __name__ == "__main__":
    main()
