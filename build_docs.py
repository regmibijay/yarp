#!/usr/bin/env python3
"""
Documentation build script for YARP.

This script provides easy commands to build and serve the documentation.
"""
import subprocess
import sys
import webbrowser
from pathlib import Path


def get_project_root():
    """Get the project root directory."""
    return Path(__file__).parent


def run_command(cmd, cwd=None):
    """Run a shell command and return the result."""
    try:
        result = subprocess.run(
            cmd, shell=True, check=True, capture_output=True, text=True, cwd=cwd
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        return False, e.stdout, e.stderr


def build_docs():
    """Build the HTML documentation."""
    project_root = get_project_root()
    docs_dir = project_root / "docs"
    venv_python = project_root / ".venv" / "bin" / "python"

    print("Building documentation...")

    # Check if virtual environment exists
    if not venv_python.exists():
        print("Error: Virtual environment not found. Please run: python -m venv .venv")
        return False

    # Build command
    cmd = f"{venv_python} -m sphinx -b html source build"
    success, stdout, stderr = run_command(cmd, cwd=docs_dir)

    if success:
        print("✅ Documentation built successfully!")
        print(f"📁 Output directory: {docs_dir / 'build'}")
        print(f"🌐 Open: {docs_dir / 'build' / 'index.html'}")
        return True
    else:
        print("❌ Documentation build failed!")
        print("STDOUT:", stdout)
        print("STDERR:", stderr)
        return False


def serve_docs(port=8000):
    """Serve the documentation with auto-reload."""
    project_root = get_project_root()
    docs_dir = project_root / "docs"
    venv_python = project_root / ".venv" / "bin" / "python"

    print(f"Starting documentation server on port {port}...")
    print("Press Ctrl+C to stop the server")

    # Use sphinx-autobuild for live reloading
    cmd = f"{venv_python} -m sphinx_autobuild source build --port {port} --host 0.0.0.0"

    try:
        subprocess.run(cmd, shell=True, check=True, cwd=docs_dir)
    except KeyboardInterrupt:
        print("\n📴 Documentation server stopped.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start server: {e}")


def clean_docs():
    """Clean the documentation build directory."""
    project_root = get_project_root()
    docs_dir = project_root / "docs"
    build_dir = docs_dir / "build"

    if build_dir.exists():
        import shutil

        shutil.rmtree(build_dir)
        print("🧹 Cleaned documentation build directory.")
    else:
        print("📁 Build directory doesn't exist, nothing to clean.")


def open_docs():
    """Open the built documentation in the default browser."""
    project_root = get_project_root()
    index_file = project_root / "docs" / "build" / "index.html"

    if index_file.exists():
        webbrowser.open(f"file://{index_file.absolute()}")
        print(f"🌐 Opened documentation in browser: {index_file}")
    else:
        print("❌ Documentation not built yet. Run 'python build_docs.py build' first.")


def main():
    """Main command handler."""
    if len(sys.argv) < 2:
        print("YARP Documentation Builder")
        print("Usage:")
        print("  python build_docs.py build     - Build HTML documentation")
        print("  python build_docs.py serve     - Build and serve with auto-reload")
        print("  python build_docs.py clean     - Clean build directory")
        print("  python build_docs.py open      - Open documentation in browser")
        return

    command = sys.argv[1].lower()

    if command == "build":
        build_docs()
    elif command == "serve":
        if not build_docs():
            return
        serve_docs()
    elif command == "clean":
        clean_docs()
    elif command == "open":
        open_docs()
    else:
        print(f"❌ Unknown command: {command}")
        print("Available commands: build, serve, clean, open")


if __name__ == "__main__":
    main()
