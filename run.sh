#!/bin/bash

set -e

VENV_PATH="./venv"
REQ_FILE="requirements.txt"

if [ ! -d "$VENV_PATH" ]; then
    echo "Virtual environment not found. Creating a new one..."
    python3 -m venv $VENV_PATH
fi

source $VENV_PATH/bin/activate

if [ $? -ne 0 ]; then
    echo "Error: Failed to activate virtual environment"
    return 1 2>/dev/null || exit 1
fi

if [ ! -f "$VENV_PATH/.requirements_installed" ] || [ "$REQ_FILE" -nt "$VENV_PATH/.requirements_installed" ]; then
    echo "Installing/Updating packages..."
    pip install -r $REQ_FILE
    touch "$VENV_PATH/.requirements_installed"
else
    echo "Requirements are up to date."
fi

export PYTHONPATH=$PYTHONPATH:$(pwd)

python src/main.py
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "Python script exited with code $EXIT_CODE"
    echo "Press any key to exit..."
    read -n 1 -s
fi

deactivate