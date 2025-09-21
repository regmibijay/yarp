# Documentation

This project uses Sphinx to generate comprehensive API documentation from docstrings.

## Building Documentation

### Prerequisites

Install documentation dependencies:

```bash
uv sync --group docs
```

### Quick Start

Use the provided build script for easy documentation management:

```bash
# Build HTML documentation
python build_docs.py build

# Build and serve with auto-reload (great for development)
python build_docs.py serve

# Clean build directory
python build_docs.py clean

# Open documentation in browser
python build_docs.py open
```

### Manual Building

You can also build documentation manually:

```bash
cd docs
sphinx-build -b html source build
```

### Live Development

For documentation development with auto-reload:

```bash
cd docs
sphinx-autobuild source build --port 8000
```

Then open http://localhost:8000 in your browser.

## Documentation Structure

- `docs/source/conf.py` - Sphinx configuration
- `docs/source/index.rst` - Main documentation page
- `docs/source/api.rst` - API reference (auto-generated)
- `docs/source/quickstart.rst` - Quick start guide
- `docs/source/installation.rst` - Installation instructions
- `docs/source/examples.rst` - Usage examples
- `docs/build/` - Generated HTML output (gitignored)

## Writing Documentation

### Docstring Style

We use Google-style docstrings with Napoleon extension. Example:

```python
def my_function(param1: str, param2: int = 10) -> bool:
    """
    Short description of the function.
    
    Longer description with more details about what the function does,
    how it works, and any important considerations.
    
    Args:
        param1 (str): Description of the first parameter.
        param2 (int, optional): Description of the second parameter.
            Defaults to 10.
    
    Returns:
        bool: Description of the return value.
        
    Raises:
        ValueError: When param1 is empty.
        TypeError: When param2 is not an integer.
        
    Example:
        >>> result = my_function("hello", 5)
        >>> print(result)
        True
    
    Note:
        Any additional notes or warnings about usage.
    """
    pass
```

### Adding New Pages

1. Create a new `.rst` file in `docs/source/`
2. Add it to the `toctree` in `index.rst`
3. Rebuild the documentation

### API Documentation

API documentation is automatically generated from docstrings using autodoc. The configuration in `conf.py` includes:

- All public members are documented
- Type hints are included in descriptions
- Cross-references to external libraries (numpy, pydantic, etc.)

## Deployment

The documentation can be deployed to various platforms:

### GitHub Pages (Automated)

**Automatic deployment is already set up!** 

The repository includes GitHub Actions workflows that automatically:
- Build documentation on every push
- Deploy to GitHub Pages when you push to master/main
- Run quality checks on documentation changes

**Setup Steps:**
1. Go to your repository Settings → Pages
2. Set Source to "GitHub Actions"
3. Push your changes - documentation will be live at: `https://regmibijay.github.io/yarp/`

For detailed setup and troubleshooting, see: [GitHub Pages Setup Guide](GITHUB_PAGES_SETUP.md)

### Read the Docs

1. Connect your repository to Read the Docs
2. The `pyproject.toml` and `docs/source/conf.py` are already configured
3. RTD will automatically build and host your documentation

## Troubleshooting

### Common Issues

1. **Import errors during build**: Make sure all dependencies are installed
2. **Missing modules**: Check that the project is in the Python path
3. **Sphinx warnings**: These are usually about missing docstrings or malformed RST

### Checking Documentation Quality

```bash
# Check for broken links
sphinx-build -b linkcheck source build

# Check for coverage (missing docstrings)
sphinx-build -b coverage source build
```