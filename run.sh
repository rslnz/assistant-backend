#!/bin/bash

VENV_PATH="./venv"

if [ ! -d "$VENV_PATH" ]; then
    echo "Virtual environment not found. Creating a new one..."
    python3 -m venv $VENV_PATH
fi

source $VENV_PATH/bin/activate

if [ $? -ne 0 ]; then
    echo "Error: Failed to activate virtual environment"
    exit 1
fi

pip install --upgrade pip
pip install -r requirements.txt --upgrade

export $(grep -v '^#' .env | xargs)
export PYTHONPATH=$PYTHONPATH:$(pwd)

uvicorn src.main:app --host $HOST --port $PORT --reload --log-level info

deactivate