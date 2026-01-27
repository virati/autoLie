#!/usr/bin/env python3
"""
lie_plots
Library for plotting Lie analyses using PyVista

@author: virati
Vineet Tiruvadi
virati@gmail.com
"""
import matplotlib.pyplot as plt
import pyvista as pv
import autograd.numpy as np

def plot_field(dyn_field, coords, normfield=False, color='', plotter=None):
    """Plot a 3D vector field using PyVista.

    Args:
        dyn_field: Vector field array with shape (3, nx, ny, nz)
        coords: Tuple of (x, y, z) meshgrid arrays
        normfield: Whether to normalize the field
        color: Color for the arrows (RGB tuple or empty string for default)
        plotter: PyVista plotter instance (creates new one if None)

    Returns:
        PyVista plotter instance
    """
    x = coords[0]
    y = coords[1]
    z = coords[2]

    # Normalize field if requested
    row_sums = dyn_field.sum(axis=0)
    if normfield:
        norm_dyn_field = 100 * dyn_field / row_sums
    else:
        norm_dyn_field = dyn_field

    # Handle infinities
    dyn_field[np.isinf(dyn_field)] = np.nan
    norm_dyn_field[np.isinf(norm_dyn_field)] = np.nan

    # Create point cloud and vector data
    points = np.column_stack([x.flatten(), y.flatten(), z.flatten()])
    vectors = np.column_stack([
        norm_dyn_field[0].flatten(),
        norm_dyn_field[1].flatten(),
        norm_dyn_field[2].flatten()
    ])

    # Create PyVista point cloud
    cloud = pv.PolyData(points)
    cloud['vectors'] = vectors

    # Create plotter if not provided
    if plotter is None:
        plotter = pv.Plotter()

    # Add arrows/glyphs
    arrows = cloud.glyph(orient='vectors', scale='vectors', factor=0.3)

    if color == '':
        plotter.add_mesh(arrows, opacity=0.9, show_scalar_bar=False)
    else:
        plotter.add_mesh(arrows, color=color, opacity=0.9, show_scalar_bar=False)

    return plotter
    
def plot_fields(dyn_field, ctrl_field, coords, normfield=False, plotter=None):
    """Plot both dynamics and control vector fields using PyVista.

    Args:
        dyn_field: Dynamics vector field array
        ctrl_field: Control vector field array
        coords: Tuple of (x, y, z) meshgrid arrays
        normfield: Whether to normalize the field
        plotter: PyVista plotter instance (creates new one if None)

    Returns:
        PyVista plotter instance
    """
    if plotter is None:
        plotter = pv.Plotter()

    # Plot dynamics field with default color
    plotter = plot_field(dyn_field, coords, normfield=normfield, plotter=plotter)

    # Plot control field with low opacity
    x, y, z = coords
    points = np.column_stack([x.flatten(), y.flatten(), z.flatten()])
    vectors = np.column_stack([
        ctrl_field[0].flatten(),
        ctrl_field[1].flatten(),
        ctrl_field[2].flatten()
    ])

    cloud = pv.PolyData(points)
    cloud['vectors'] = vectors
    arrows = cloud.glyph(orient='vectors', scale='vectors', factor=0.3)
    plotter.add_mesh(arrows, opacity=0.1, show_scalar_bar=False)

    return plotter


# Plotting methods here
# this here plots the potential grid arising from the function itself
def plot_LD(func, normalize=False, dens=10, plotter=None):
    """Plot Lie derivative as vector field.

    Args:
        func: Function to evaluate
        normalize: Whether to normalize vectors
        dens: Grid density
        plotter: PyVista plotter instance (creates new one if None)

    Returns:
        PyVista plotter instance
    """
    x, y, z = gen_meshgrid(dens)

    potgrid = func([x, y, z], [0])

    if normalize:
        for xx in range(dens):
            for yy in range(dens):
                for zz in range(dens):
                    norm = np.linalg.norm(potgrid[:, xx, yy, zz])
                    if norm > 0:
                        potgrid[:, xx, yy, zz] = potgrid[:, xx, yy, zz] / norm
    else:
        potgrid = potgrid / 500

    potgrid = 10 * potgrid

    # Create point cloud
    points = np.column_stack([x.flatten(), y.flatten(), z.flatten()])
    vectors = np.column_stack([
        potgrid[0].flatten(),
        potgrid[1].flatten(),
        potgrid[2].flatten()
    ])

    cloud = pv.PolyData(points)
    cloud['vectors'] = vectors

    if plotter is None:
        plotter = pv.Plotter()

    arrows = cloud.glyph(orient='vectors', scale='vectors', factor=0.1)
    plotter.add_mesh(arrows, color=(0.0, 1.0, 0.0), opacity=1.0, show_scalar_bar=False)

    return plotter

# here we find the lie for the Laplacian
def plot_Ldot(func, D=None, normed=True, plotter=None):
    """Plot Lie dot product as scalar point cloud.

    Args:
        func: Function to evaluate
        D: Optional parameter for function
        normed: Whether to use log-normalized values
        plotter: PyVista plotter instance (creates new one if None)

    Returns:
        PyVista plotter instance
    """
    x, y, z = gen_meshgrid(dens=20)

    if D is None:
        potgrid = func([x, y, z])
    else:
        potgrid = func([x, y, z, D])

    # Sum across first axis
    scalar_field = np.sum(potgrid, axis=0)

    # Create point cloud
    points = np.column_stack([x.flatten(), y.flatten(), z.flatten()])
    cloud = pv.PolyData(points)

    if normed:
        # Use log scale for visualization
        scalars = np.abs(np.log10(scalar_field.flatten() + 1e-10))
        cloud['scalars'] = scalars
        point_size = 10.0
    else:
        cloud['scalars'] = scalar_field.flatten()
        point_size = 5.0

    if plotter is None:
        plotter = pv.Plotter()

    plotter.add_mesh(cloud, point_size=point_size, render_points_as_spheres=True, opacity=1.0)

    return plotter

# Misc stuff here

def gen_meshgrid(dens=20):
    """Generate a 3D meshgrid for visualization.

    Args:
        dens: Grid density (number of points per dimension)

    Returns:
        Tuple of (x, y, z) meshgrid arrays
    """
    x_ = np.linspace(-10, 10, dens)
    y_ = np.linspace(-10, 10, dens)
    z_ = np.linspace(-10, 10, dens)

    x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')

    return x, y, z


def gen_arbgrid(dims=4, dens=20):
    """Generate arbitrary-dimension grid.

    Args:
        dims: Number of dimensions
        dens: Grid density

    Returns:
        Grid array
    """
    X = np.array([np.linspace(-10, 10, dens) for aa in np.arange(dims)])
    grid = np.meshgrid(X, indexing='ij')

    return grid
