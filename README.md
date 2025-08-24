# Machine Learning Project Template

A cookie-cutter template for machine learning projects with PyTorch, scikit-learn
and transformers.


Getting started in a single line:
```bash
cookiecutter https://github.com/EchoStatements/Machine-Learning-Project-Template
```

## What is this?

The repo is a cookie cutter for quickly generating the environment and config for
a quick and lightweight repository to rapidly test ideas using standard ML libraries.

This template is primarily designed for my own personal use, though I'm happy to take
contributions from others if they find it useful and have ideas for improvements.

## Usage

### Prerequisites

- Python 3.7+
- cookiecutter (`pip install cookiecutter`)

### Creating a New Project

```bash
cookiecutter https://github.com/EchoStatements/Machine-Learning-Project-Template
```

You'll be prompted to provide information about your project:

- `project_name`: Name of your project
- `project_slug`: Slug for your project (derived from project_name by default)
- `project_description`: Brief description of your project
- `author_name`: Your name
- `author_email`: Your email
- `python_version`: Minimum Python version required (default: 3.12)
- `open_source_license`: License for your project (MIT, BSD-3-Clause, or no license)

### Project Structure

The generated project will have the following structure:

```
your_project/
├── data/               # Data files
├── models/             # Saved models
├── notebooks/          # Jupyter notebooks
│   └── ml_example.ipynb # Example notebook
├── src/                # Source code
│   └── your_project/
│       ├── __init__.py
│       └── ml_example.py # Example ML script
├── tests/              # Unit tests
├── pyproject.toml
└── README.md
```

### Getting Started

After creating your project:

1. Create and activate a virtual environment:
   ```bash
   cd your_project
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install the package and development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

3. Set up pre-commit hooks:
   ```bash
   pre-commit install
   ```

4. Run the example script:
   ```bash
   python src/your_project/ml_example.py
   ```

## License

MIT
