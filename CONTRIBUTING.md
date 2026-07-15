# Contributing to AuraWear Analysis

Thank you for your interest in contributing. Please read the guidelines below before opening issues or pull requests.

## Scope

This is a research prototype. Contributions are welcome in the following areas:

- Bug fixes
- Documentation improvements
- Additional demo catalog items (synthetic data only — no third-party dataset derivatives)
- Improved graceful degradation when LLM or face parsing model is unavailable
- Performance improvements that do not change output semantics
- Test coverage

## Before Contributing

1. **Check existing issues** — your idea may already be discussed.
2. **Open an issue first** for significant changes, to align on scope before implementing.
3. **Do not include** DeepFashion-MultiModal files, CelebAMask-HQ weights, real selfies, participant data, or any API keys in pull requests.

## Code Style

- Python 3.10+
- Follow the conventions in the existing codebase (type hints, dataclasses, docstrings on public functions)
- No new dependencies unless clearly justified in the PR description

## Pull Request Process

1. Fork the repository and create a feature branch.
2. Make changes, ensure the synthetic demo runs: `python app_gradio.py`
3. Open a pull request with a clear description of what was changed and why.

## License

By submitting a contribution, you agree that your work will be licensed under the project's [AGPL-3.0-or-later license](LICENSE).

## Attribution

See [AUTHORS.md](AUTHORS.md) and [CONTRIBUTION.md](CONTRIBUTION.md) for the project's authorship history.
