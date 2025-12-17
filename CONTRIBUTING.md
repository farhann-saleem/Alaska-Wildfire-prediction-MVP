# Contributing to Alaska Wildfire Prediction

Thank you for your interest in contributing to the Alaska Wildfire Prediction project! This project is part of **Google Summer of Code 2026** with the University of Alaska Anchorage.

## 🌟 Ways to Contribute

- **Code contributions:** Implement new features, fix bugs, improve performance
- **Documentation:** Enhance docs, write tutorials, improve README
- **Testing:** Add unit tests, report bugs, suggest test cases
- **Research:** Experiment with new models, datasets, or techniques
- **Community:** Answer questions, review PRs, participate in discussions

---

## 🚀 Getting Started

### Development Setup

1. **Fork and clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/wildfire-prediction-mvp.git
   cd wildfire-prediction-mvp
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: .\venv\Scripts\Activate.ps1
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Create a feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

---

## 📝 Contribution Guidelines

### Code Style

- Follow [PEP 8](https://peps.python.org/pep-0008/) for Python code
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Keep functions focused and modular (< 50 lines ideally)

**Example:**
```python
def extract_patch(image_array, x, y, patch_size=64):
    """
    Extract a square patch from a larger image array.
    
    Args:
        image_array (np.ndarray): Input image (H, W, C)
        x (int): Top-left x coordinate
        y (int): Top-left y coordinate
        patch_size (int): Patch dimension (default: 64)
    
    Returns:
        np.ndarray: Extracted patch (patch_size, patch_size, C)
    """
    return image_array[y:y+patch_size, x:x+patch_size, :]
```

### Git Workflow

1. **Commit messages:** Use clear, descriptive commit messages
   ```bash
   # Good
   git commit -m "Add Sentinel-1 SAR data loader"
   
   # Bad
   git commit -m "fixed stuff"
   ```

2. **Keep commits focused:** One logical change per commit

3. **Pull before push:** Always sync with main branch
   ```bash
   git pull origin main
   git push origin feature/your-feature-name
   ```

### Pull Request Process

1. **Update documentation** if you change functionality
2. **Add tests** for new features
3. **Ensure all tests pass** before submitting
4. **Fill out the PR template** completely
5. **Reference related issues** (e.g., "Fixes #42")

**PR Title Format:**
```
[Component] Brief description

Examples:
[Data Pipeline] Add multi-temporal Sentinel-2 loader
[Model] Implement CNN-LSTM architecture
[Docs] Update installation guide for Windows
```

---

## 🧪 Testing

### Running Tests

```bash
# Install pytest
pip install pytest

# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_data_pipeline.py -v
```

### Writing Tests

- Place tests in `tests/` directory
- Name test files as `test_*.py`
- Name test functions as `test_*`

**Example:**
```python
# tests/test_models.py
import pytest
from src.models.wildfire_cnn import create_enhanced_cnn

def test_model_creation():
    """Test that model can be instantiated."""
    model = create_enhanced_cnn(input_shape=(64, 64, 3))
    assert model is not None
    assert len(model.layers) > 0
```

---

## 📚 Documentation

### Adding Documentation

- **Code documentation:** Use docstrings (Google style)
- **User documentation:** Update relevant `.md` files in `docs/`
- **README:** Keep README concise, link to detailed docs

### Documentation Files

- `README.md` - Project overview and quick start
- `docs/architecture.md` - System design
- `docs/data-pipeline.md` - Data processing details
- `docs/model-training.md` - ML methodology

---

## 🐛 Reporting Bugs

### Before Submitting

1. **Search existing issues** to avoid duplicates
2. **Test on latest version** from main branch
3. **Gather information** (error messages, logs, environment)

### Bug Report Template

```markdown
**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Run command '...'
2. With input '...'
3. See error

**Expected behavior**
What you expected to happen.

**Environment:**
- OS: [e.g., Windows 11, Ubuntu 22.04]
- Python version: [e.g., 3.10.5]
- TensorFlow version: [e.g., 2.13.0]

**Additional context**
Error logs, screenshots, etc.
```

---

## 💡 Feature Requests

We welcome ideas for new features! Please:

1. **Check existing issues** to see if someone already suggested it
2. **Explain the use case** - why is this feature needed?
3. **Describe the solution** - how should it work?
4. **Consider alternatives** - are there other approaches?

---

## 🎓 GSoC Contributors

### For GSoC Applicants

If you're applying for **GSoC 2026**, please:

1. **Introduce yourself** in [GitHub Discussions](https://github.com/uaanchorage/GSoC/discussions)
2. **Review the rubrics** at [Alaska GSoC Rubrics](https://github.com/uaanchorage/GSoC/blob/main/Rubrics.md)
3. **Make meaningful contributions** - quality over quantity
4. **Engage with the community** - help other contributors, review PRs

### GSoC Project Ideas

- **Multi-modal fusion:** Integrate Sentinel-1 SAR and weather data
- **Temporal modeling:** Implement CNN-LSTM for time-series prediction
- **Web dashboard:** Create interactive visualization for predictions
- **Data augmentation:** Develop domain-specific augmentation strategies
- **Model optimization:** Improve inference speed and deployment readiness

---

## 🤝 Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive environment for all contributors, regardless of experience level, background, or identity.

### Expected Behavior

- Be respectful and considerate
- Provide constructive feedback
- Focus on what's best for the project and community
- Show empathy towards other contributors

### Unacceptable Behavior

- Harassment, discrimination, or personal attacks
- Trolling or inflammatory comments
- Publishing others' private information
- Other conduct that could be considered inappropriate

---

## 📧 Contact & Communication

### Primary Channels

- **GitHub Issues:** Bug reports and feature requests
- **GitHub Discussions:** Questions, ideas, and general discussion
- **GSoC Forums:** [Alaska GSoC Discussions](https://github.com/uaanchorage/GSoC/discussions)

### Response Time

- Issues and PRs are typically reviewed within 2-3 business days
- For urgent matters, tag the issue with `priority: high`

---

## 🏆 Recognition

Contributors will be:
- Listed in the project README
- Mentioned in release notes for significant contributions
- Acknowledged in academic publications (if applicable)

---

## 📄 License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).

---

Thank you for helping make Alaska safer from wildfires! 🔥🌲
