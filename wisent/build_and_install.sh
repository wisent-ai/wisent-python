#!/bin/bash
# Build and install the Wisent package

set -e  # Exit on error

# Clean up previous builds
echo "Cleaning up previous builds..."
rm -rf build/ dist/ *.egg-info/

# Install build dependencies
echo "Installing build dependencies..."
pip install --upgrade pip
pip install --upgrade build twine wheel

# Build the package
echo "Building the package..."
python -m build

# Install the package locally
echo "Installing the package locally..."
pip install --force-reinstall dist/*.whl

echo "Build and installation complete!"
echo "To publish to PyPI, run: python -m twine upload dist/*" 