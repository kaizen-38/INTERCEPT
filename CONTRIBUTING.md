# Contributing to INTERCEPT

Thank you for your interest in contributing to INTERCEPT! This document provides guidelines for contributing to this research project.

## Getting Started

### Development Setup

1. Clone the repository:
```bash
git clone https://github.com/kaizen-38/INTERCEPT.git
cd INTERCEPT
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install development dependencies:
```bash
pip install -e ".[all]"
```

4. Run tests to verify setup:
```bash
pytest tests/ -v
```

## Code Style

- Follow [PEP 8](https://pep8.org/) conventions
- Use [Black](https://github.com/psf/black) for formatting: `black src/`
- Use type hints for all function signatures
- Write docstrings for all public functions/classes (Google style)

### Example Docstring

```python
def my_function(param1: int, param2: str) -> bool:
    """Short description of the function.
    
    Longer description if needed, explaining the behavior
    in more detail.
    
    Args:
        param1: Description of param1
        param2: Description of param2
    
    Returns:
        Description of return value
    
    Raises:
        ValueError: When param1 is negative
    """
```

## Project Structure

```
INTERCEPT/
├── src/                    # Source code
│   ├── intercept_env.py    # Core environment
│   ├── intercept_grpo.py   # Policy and training
│   ├── intercept_baselines.py  # Baseline strategies
│   └── ...
├── tests/                  # Unit tests
├── notebooks/              # Jupyter notebooks
├── data/                   # Datasets
├── results/                # Training outputs
└── figures/                # Generated visualizations
```

## Making Changes

### For Bug Fixes

1. Create an issue describing the bug
2. Fork the repository
3. Create a branch: `git checkout -b fix/issue-description`
4. Write a test that reproduces the bug
5. Fix the bug
6. Verify the test passes
7. Submit a pull request

### For New Features

1. Open an issue to discuss the feature
2. Fork the repository
3. Create a branch: `git checkout -b feature/feature-name`
4. Implement the feature with tests
5. Update documentation if needed
6. Submit a pull request

### For Research Extensions

If you're extending INTERCEPT for your research:

1. **New Baselines**: Add to `src/intercept_baselines.py` following the `BaselineStrategy` interface
2. **New Networks**: Add loaders to `src/network_datasets.py`
3. **New Policies**: Extend `TemporalGRPOPolicy` or create new architectures
4. **New Environments**: Modify `IndependentCascadeEnv` or create variants

## Running Experiments

### Training

```bash
python -m src.train_intercept
```

### Evaluation

```bash
python -m src.evaluate_intercept \
    --checkpoint results/intercept_grpo_*/checkpoints/checkpoint_group_0100.pt
```

### Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_environment.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Submitting Pull Requests

1. Ensure all tests pass
2. Update documentation for any changed functionality
3. Add entries to README.md if adding new features
4. Keep commits atomic and well-described
5. Reference any related issues

## Research Collaboration

If you're using INTERCEPT in your research and would like to collaborate:

- Open an issue tagged `[Research]`
- Share your experimental setup and findings
- Consider contributing back improvements

## Citation

If you use INTERCEPT in your research, please cite:

```bibtex
@software{intercept2025,
  title={INTERCEPT: Intervention Reinforcement Control for Epidemic Prevention Transmission},
  author={kaizen-38},
  year={2025},
  url={https://github.com/kaizen-38/INTERCEPT}
}
```

## Questions?

- Open a GitHub issue for bugs or feature requests
- Tag issues with appropriate labels (`bug`, `enhancement`, `question`)

Thank you for contributing!

