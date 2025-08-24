#!/usr/bin/env python
"""Post-generation script for the cookiecutter template."""

import os
import shutil
import subprocess
from pathlib import Path


def create_empty_dirs():
    """Create empty directories that might not be included in version control."""
    dirs = [
        "data",
        "models",
        "figures",
    ]

    for d in dirs:
        os.makedirs(d, exist_ok=True)
        # Create a .gitkeep file to ensure the directory is included in version control
        with open(os.path.join(d, ".gitkeep"), "w") as f:
            f.write("")


def create_gitignore():
    """Create a .gitignore file with common patterns for ML projects."""
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
ENV/
env/

# Jupyter Notebook
.ipynb_checkpoints

# Data files
*.csv
*.tsv
*.xlsx
*.db
*.sqlite3

# Model files
*.pt
*.pth
*.h5
*.pkl
*.joblib

# Logs
logs/
*.log

# IDE
.idea/
.vscode/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
"""

    with open(".gitignore", "w") as f:
        f.write(gitignore_content)


def main():
    """Run all post-generation tasks."""
    print("Running post-generation tasks...")

    create_empty_dirs()
    create_gitignore()

    print("Post-generation tasks completed successfully!")


if __name__ == "__main__":
    main()
