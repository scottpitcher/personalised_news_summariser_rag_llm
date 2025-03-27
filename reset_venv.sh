#!/bin/zsh

# Deactivate virtual environment if active
if [[ "$VIRTUAL_ENV" != "" ]]; then
    deactivate
    echo "Deactivated virtual environment."
fi

echo "After deactivate, shell: $SHELL"
# Remove existing virtual environment
rm -rf .venv
echo "Old venv removed."

# Create a new virtual environment
python3.10 -m venv .venv
echo "New venv created."

# Activate the virtual environment
source .venv/bin/activate || . .venv/bin/activate
echo "Virtual environment activated."

# Upgrade pip
pip install --upgrade pip
echo "Pip upgraded."

# Restore packages if requirements.txt exists
if [[ -f "requirements.txt" ]]; then
    pip install -r requirements.txt
    echo "Packages installed from requirements.txt."
else
    echo "requirements.txt not found, skipping package installation."
fi

# Indicate when the process is finished
echo "Virtual environment created!"