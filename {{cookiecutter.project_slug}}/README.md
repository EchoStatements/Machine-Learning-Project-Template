# {{ cookiecutter.project_name }}

{{ cookiecutter.project_description }}

## Project Structure

```
{{ cookiecutter.project_slug }}/
├── {{ cookiecutter.project_slug }}/  # Source code
│   ├── __init__.py
│   └── ml_example.py   # Example ML script
├── data/               # Data files
├── models/             # Saved models
├── notebooks/          # Jupyter notebooks
│   └── ml_example.ipynb # Example notebook
├── tests/              # Unit tests
├── pyproject.toml
└── README.md
```

## Installation

### Prerequisites

- Python {{ cookiecutter.python_version }} or higher

### Setup

1. Clone the repository:

```bash
git clone <repository-url>
cd {{ cookiecutter.project_slug }}
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the package and development dependencies:

```bash
pip install -e ".[dev]"
```

4. Set up pre-commit hooks:

```bash
pre-commit install
```

## Usage

### Running Examples

The `{{ cookiecutter.project_slug }}` directory contains a script using some
of the default libraries that can be used to test that the project has been
initialised correctly. This script can be run using the command

```bash
python {{ cookiecutter.project_slug }}/ml_example.py
```

### Jupyter Notebooks

The `notebooks` directory contains Jupyter notebooks for interactive development and visualization:

```bash
jupyter notebook notebooks/
```

## Development

### Running Tests

```bash
pytest
```

### Code Quality

This project uses pre-commit hooks to enforce code quality:

- **ruff**: For linting and formatting Python code
- **typos**: For spell checking (excluding .ipynb files)

The maximum complexity for functions is set to 10 using McCabe complexity metric.

## License

{{ cookiecutter.open_source_license }}
