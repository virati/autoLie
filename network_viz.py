#!/usr/bin/env python3
"""
Network visualization example using PyVista

@author: virati
Vineet Tiruvadi
virati@gmail.com
"""

import networkx as nx
import numpy as np
import pyvista as pv
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt

# some graphs to try
# H=nx.krackhardt_kite_graph()
# H=nx.Graph();H.add_edge('a','b');H.add_edge('a','c');H.add_edge('a','d')
# H=nx.grid_2d_graph(4,5)
# H=nx.cycle_graph(20)
H = nx.erdos_renyi_graph(20, 0.3)

L = np.abs(nx.laplacian_matrix(H)).todense()

theta = np.random.multivariate_normal(12 * np.ones(L.shape[0]), L, 10000).T
# %%
# plt.plot(y)

t = np.linspace(0, 10, 10000)
y = np.zeros((16, 10000))
for ii in range(16):
    y[ii, :] = np.sin(2 * np.pi * theta[ii, :] * t)

plt.plot(y[:, 0:1000].T)

# %%
# Now do empiric variance
empir_var = np.zeros((16, 16))
for ii in range(16):
    for jj in range(16):
        empir_var[ii, jj] = np.cov(np.vstack((y[ii, :], y[jj, :])))[0, 1]


plt.figure()
plt.subplot(211)
plt.title("Estimated Laplacian through Covar")
plt.imshow(empir_var > 0.01)
plt.subplot(212)
plt.imshow(L)
plt.title("True Laplacian")


# %%
def plot_graph():
    G = nx.erdos_renyi_graph(7, 0.2)
    plt.figure()
    nx.draw(G)


plot_graph()
# %%


def render_graph(H, read, write, save_path="pyvista_spring.png"):
    """Render a graph in 3D using PyVista.

    Args:
        H: NetworkX graph
        read: Boolean array marking read nodes
        write: Boolean array marking write nodes
        save_path: Path to save screenshot

    Returns:
        PyVista plotter instance
    """
    # reorder nodes from 0,len(G)-1
    G = nx.convert_node_labels_to_integers(H)
    # 3d spring layout
    pos = nx.spring_layout(G, dim=3)
    # numpy array of x,y,z positions in sorted node order
    xyz = np.array([pos[v] for v in sorted(G)])
    # scalar colors
    scalars = np.array(list(G.nodes())) + 5

    # Create plotter with black background
    plotter = pv.Plotter()
    plotter.set_background("black")

    # Add all nodes as spheres
    nodes_cloud = pv.PolyData(xyz)
    nodes_cloud["scalars"] = scalars
    spheres = nodes_cloud.glyph(geom=pv.Sphere(radius=0.05), scale=False)
    plotter.add_mesh(spheres, cmap="Blues", show_scalar_bar=False)

    # Add read nodes (green)
    if np.any(read):
        read_cloud = pv.PolyData(xyz[read])
        read_spheres = read_cloud.glyph(geom=pv.Sphere(radius=0.05), scale=False)
        plotter.add_mesh(read_spheres, color="green", show_scalar_bar=False)

    # Add write nodes (red)
    if np.any(write):
        write_cloud = pv.PolyData(xyz[write])
        write_spheres = write_cloud.glyph(geom=pv.Sphere(radius=0.05), scale=False)
        plotter.add_mesh(write_spheres, color="red", show_scalar_bar=False)

    # Add edges as tubes
    for edge in G.edges():
        line = pv.Line(xyz[edge[0]], xyz[edge[1]])
        tube = line.tube(radius=0.01)
        plotter.add_mesh(tube, color=(0.8, 0.8, 0.8), opacity=0.1)

    # Save screenshot
    plotter.show()
    plotter.screenshot(save_path)

    return plotter


read = np.zeros((20,), dtype=bool)
write = np.zeros((20,), dtype=bool)
read[2] = True
write[5] = True

plotter = render_graph(H, read, write)
# To show interactively, uncomment:
# plotter.show()
