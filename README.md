# Machine Learning Project Template

A cookie-cutter template for machine learning projects with PyTorch, scikit-learn, and transformers.


To get started:
```bash
cookiecutter https://github.com/yourusername/uv-machine-learning-template
```

## What is this?

The repo is a cookie cutter for quickly generating the environment and config for
a quick and lightweight repository to rapidly test ideas using standard ML libraries.

## Usage

### Prerequisites

- Python 3.7+
- cookiecutter (`pip install cookiecutter`)

### Creating a New Project

```bash
cookiecutter https://github.com/yourusername/uv-machine-learning-template
```

You'll be prompted to provide information about your project:

- `project_name`: Name of your project
- `project_slug`: Slug for your project (derived from project_name by default)
- `project_description`: Brief description of your project
- `author_name`: Your name
- `author_email`: Your email
- `python_version`: Minimum Python version required (default: 3.12)
- `open_source_license`: License for your project (MIT, BSD-3-Clause, or no license)
- `include_jupyter_support`: Whether to include Jupyter notebook support

### Project Structure

The generated project will have the following structure:

```
your_project/
├── data/               # Data files
├── models/             # Saved models
├── notebooks/          # Jupyter notebooks
├── src/                # Source code
│   └── your_project/
│       ├── __init__.py
│       ├── data.py     # Data loading and processing
│       ├── model.py    # Model definitions
│       └── train.py    # Training utilities
├── examples/           # Example scripts
├── tests/              # Unit tests
├── .pre-commit-config.yaml
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
   python examples/ml_example.py
   ```

## Customization

The template provides a solid foundation for machine learning projects, but you can customize it to fit your specific needs:

- Add or remove dependencies in `pyproject.toml`
- Modify the model architecture in `src/your_project/model.py`
- Customize data loading in `src/your_project/data.py`
- Add new modules for specific functionality

## License

MIT
