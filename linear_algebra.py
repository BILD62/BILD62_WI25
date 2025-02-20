import numpy as np
import matplotlib.pyplot as plt


# @title Plotting functions
def visualize_vectors(v, v_unit=None):
    """ Plots a 2D vector and the corresponding unit vector

    Args:
        v (ndarray): array of size (2,) with the vector coordinates
        v_unit (ndarray): array of size (2, ) with the unit vector coordinates

      """
    fig, ax = plt.subplots(figsize=(5,5))

    # Set up plot aesthetics
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.set(xlim = [-6, 6], ylim = [-6, 6])
    ax.grid(True, alpha=.4, linewidth=1, zorder=0)

    # Plot vectors
    v_arr = ax.arrow(0, 0, v[0], v[1], width=0.08, color='#648FFF', length_includes_head = True, zorder = 2);
    
    if v_unit is not None:
        v_unit_arr = ax.arrow(0, 0, v_unit[0], v_unit[1], width=0.08, color='#DC267F', length_includes_head = True, zorder = 3);
        ax.set(xlim = [-4, 4], ylim = [-4, 4]);
        # Add legend
        leg = ax.legend([v_arr, v_unit_arr], [r"Vector 1", r"Vector 2"], handlelength = 0, fontsize = 10, loc = 'upper left')
        for handle, label in zip(leg.legendHandles, leg.texts):
            label.set_color(handle.get_facecolor())
            handle.set_visible(False)

from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)
