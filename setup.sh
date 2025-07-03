#!/bin/bash

# Smart setup/activation script for optimal execution electricity markets project
# 
# IMPORTANT: This script must be sourced, not executed directly!
# Run with: source setup.sh
# Or:       . setup.sh
#
# DO NOT run with: ./setup.sh (this won't activate the environment)

echo "Checking environment setup..."

# Load Python module
module load Python/3.11.5-GCCcore-13.2.0

# Check if poetry is installed
if ! command -v poetry &> /dev/null; then
    echo "Poetry not found. Installing poetry..."
    curl -sSL https://install.python-poetry.org | python3 -
    export PATH="$HOME/.local/bin:$PATH"
else
    echo "Poetry found."
    export PATH="$HOME/.local/bin:$PATH"
fi

# Check if dependencies are installed
if [ ! -d "$(poetry env info --path 2>/dev/null)" ] || ! poetry check &> /dev/null; then
    echo "Installing project dependencies..."
    poetry install
else
    echo "Dependencies already installed."
fi

echo "Activating poetry environment..."
source $(poetry env info --path)/bin/activate

echo "Environment ready!"
echo "Python location: $(which python)"
echo "To deactivate later, run: deactivate"
