# Modernization Guide

This document describes the modernization changes made to the autoLie codebase.

## Key Changes

### 1. PyVista Replaces Mayavi

**Why**: PyVista is the modern standard for 3D scientific visualization in Python with:
- Active development and regular updates
- Better API design and documentation
- No heavy VTK GUI dependencies
- Better integration with modern Python tooling
- Support for both interactive and headless rendering

**Migration**:
- All `mayavi.mlab` calls replaced with `pyvista` equivalents
- `quiver3d()` → `pv.PolyData()` with `.glyph()` for vector fields
- `points3d()` → `pv.PolyData()` with sphere glyphs for point clouds
- `plot3d()` → `pv.PolyData()` with `.tube()` for trajectories
- All plotting functions now return a `plotter` object for composability

**Updated Files**:
- `src/lie_plots.py` - Core plotting library
- `network_viz.py` - Network visualization
- `examples/field_interact.py` - Example scripts

### 2. Modern Python Practices

**Removed**:
- `# -*- coding: utf-8 -*-` (UTF-8 is default in Python 3)
- `from __future__ import division` (not needed in Python 3)

**Updated**:
- `np.bool` → `bool` (deprecated in NumPy 1.20+)
- Added docstrings to all public functions
- Improved code formatting and consistency

### 3. Modern Package Management

**pyproject.toml**:
- Full project metadata (name, version, authors, description)
- Explicit dependencies with version constraints
- Development dependencies in `[project.optional-dependencies]`
- Ruff configuration for linting and formatting
- Uses hatchling as build backend

**Dependencies**:
```toml
dependencies = [
    "autograd>=1.6.2",
    "numpy>=1.26.0,<2.0",
    "scipy>=1.11.0",
    "matplotlib>=3.8.0",
    "networkx>=3.2.0",
    "pyvista>=0.43.0",
]
```

### 4. Code Quality Tools

**Ruff** configured for:
- Code linting (pycodestyle, pyflakes, pyupgrade, flake8-bugbear)
- Auto-formatting
- Import sorting
- Python 3.12+ compatibility checks

## Installation

### New Way (Recommended)
```bash
# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

### Legacy Way
```bash
pip install autograd numpy scipy matplotlib networkx pyvista
```

## Usage Changes

### Before (Mayavi)
```python
from mayavi import mlab
import lie_plots as lip

# Create visualization
mlab.figure(bgcolor=(0, 0, 0))
lip.plot_field(dyn_field, coords)
lip.plot_LD(y_dot)
mlab.show()
```

### After (PyVista)
```python
import pyvista as pv
import lie_plots as lip

# Create visualization
plotter = pv.Plotter()
plotter.set_background('black')
plotter = lip.plot_field(dyn_field, coords, plotter=plotter)
plotter = lip.plot_LD(y_dot, plotter=plotter)
plotter.show()
```

**Key Differences**:
1. Explicit plotter management (better control)
2. Functions return plotter for chaining
3. Optional plotter parameter (creates new one if None)
4. `plotter.show()` instead of `mlab.show()`

## Testing

Run the modernized examples:
```bash
# Network visualization
python network_viz.py

# Field interaction examples
python examples/field_interact.py
```

## Development

### Linting and Formatting
```bash
# Check code quality
ruff check .

# Auto-fix issues
ruff check --fix .

# Format code
ruff format .
```

### Running Examples
All examples have been updated to work with PyVista. The main scripts now:
1. Return plotter objects
2. Have commented-out `.show()` calls to prevent blocking
3. Include proper docstrings

## Backwards Compatibility

The core Lie algebra functionality (`lie_lib.py`, `dyn_lib.py`, `op_plus.py`) remains unchanged in terms of API. Only visualization code has changed.

If you have existing code using Mayavi, you'll need to update:
1. Import statements (`mayavi.mlab` → `pyvista`)
2. Plotting calls (see usage changes above)
3. Figure management (explicit plotter instead of global state)

## Benefits

1. **Better Performance**: PyVista uses modern VTK rendering
2. **Easier Installation**: No complex VTK/Qt dependencies
3. **Better Documentation**: PyVista has excellent docs and examples
4. **Modern Tooling**: Works well with Jupyter, VS Code, etc.
5. **Active Development**: Regular updates and bug fixes
6. **Headless Rendering**: Easy to generate images without display

## Known Issues

None currently. If you encounter issues, please check:
1. Python version >= 3.12
2. All dependencies installed: `pip install -e ".[dev]"`
3. Virtual environment is activated
