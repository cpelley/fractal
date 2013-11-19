# The MIT License (MIT)
#
# Copyright (c) 2013 cpelley
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import numpy as np
import numpy.lib.stride_tricks as stride
PLOT_UTIL = None
try:
    from mayavi import mlab
    PLOT_UTIL = 'mayavi'
except ImportError:
    import mpl_toolkits.mplot3d
    import matplotlib.pyplot as plt
    PLOT_UTIL = 'matplotlib'


def window_views(xsize=3, ysize=3, xstep=1, ystep=1):
    """
    Generate view ndarray of the grid.

    Kwargs:

    * xsize (int):
        Window dim-0 size.
    * ysize (int):
        Window dim-1 size.
    * xstep (int):
        Window dim-0 step, corresponding to window center.
    * ystep(int):
        Window dim-1 step, corresponding to window center.

    Returns:
        Numpy ndarray which represents a view of the grid.

    .. note::

    This function is not currently in use.

    """
    strides = (grid.strides[0] * xstep,
               grid.strides[1] * ystep,
               grid.strides[0],
               grid.strides[1])
    window = (((grid.shape[0] - 1) / (xsize - 1)),
              ((grid.shape[1] - 1) / (ysize - 1)),
              xsize,
              ysize)
    all_windows = stride.as_strided(grid, window, strides)
    return all_windows


def subdivide_grid(iteration, size):
    """
    Determine step size from subdivision number.

    Args:

    * iteration (int):
        Subdivision iteration number.  The relationship between dimension and
        iteration is as follows:
        dim_size_x, dim_size_y = (2**n + 1), (2**n + 1)

    Returns:
        Dictionary with size of window and the step sizes toward their center
        points information.

    """
    xsize = (size[0] - 1) / (2 ** iteration)
    ysize = (size[1] - 1) / (2 ** iteration)
    xstep = (xsize) / 2
    ystep = (ysize) / 2

    # Double checking to ensure compliance with algorithm.
    if xstep != ystep:
        raise RuntimeError('step sizes do not match')
    if xsize != ysize:
        raise RuntimeError('window dimensions not equal')

    return xsize, ysize, xstep, ystep


def perturbation(constant, length=None):
    """
    Provides a random number in the range [-constant, +constant].

    Args:

    * constant (float):
        Defines the random value range.
    * length (int)
        Number of random numbers to return.

    Returns:
    ndarray of random numbers in the range [-constant, +constant].

    """
    return np.random.rand() * 2 * constant - constant


def square_det(xstep, ystep, step_x, step_y, size, wrap=True):
    """
    Return index of all edge midpoints of surrounding point.

    With 'O' representing the current location in the grid, 'o' represents
    point in the immediate neighbourhood that are not used, while those marked
    with 'X' have their location returned:

    o X o
    X O X
    o X o

    Args:

    * xstep (int):
        The number of indexes between points considered in 'x' (corresponds
        to an iteratively subdivided grid).
    * ystep (int):
        The number of indexes between points considered in 'y' (corresponds
        to an iteratively subdivided grid).
    * step_x (int):
        x Location of our point with respect to the entire grid.
    * step_y (int):
        y Location of our point with respect to the entire grid.

    Kwargs:

    * wrap (bool):
        Performs wrap-around for neighbours on the grid edge if True.  Default
        is to wrap-around.

    Returns:
        x index ndarray and y index ndarray representing the location of each
        surrounding point in the grid.

    """
    # Performs wraparound by default.
    center_x = np.array([step_x - xstep,
                         step_x,
                         step_x + xstep,
                         step_x])
    center_y = np.array([step_y,
                         step_y + ystep,
                         step_y,
                         step_y - ystep])
    if wrap:
        # Sort our wrapped indexes.
        center_x[center_x < 0] -= 1
        center_y[center_y < 0] -= 1
        center_x[center_x > size[0] - 1] = (
            center_x[center_x > size[0] - 1] * (- 1) + (size[0] - 2))
        center_y[center_y > size[1] - 1] = (
            center_y[center_y > size[1] - 1] * (- 1) + (size[1] - 2))
    else:
        # Remove elements being wrapped.
        mask = ((center_x >= 0) & (center_x < size[0]) &
                (center_y >= 0) & (center_y < size[1]))
        center_x = center_x[mask]
        center_y = center_y[mask]

    return center_y, center_x


def midpoint_sd(size, scale=0.5, roughness=1.0, random_corner=True):
    """
    Midpoint displacement algorithm.

    The dimaond-square version is implemented.

    Args:

    * size (tuple):
        (x, y) representing the resulting grid size.  This also indirectly
        determines the number of subdivision iterations for the algorithm.
        This is determined by the following relation:
        size_0, size_1 = (2**s + 1), (2**s + 1) where 's' is the number of
        iterations.  Rearranging for s = log_2(size_x - 1).  The dimension of
        the grid must be equal and must abide by, 2**n + 1 where n in Z.

    Kwargs:

    * scale (float):
        Determines the scale of the resulting terrain.  This value has no
        impact on the behaviour of the algorithm.
    * roughtness (float):
        Roughness constant which determines the roughness of the resulting
        heightmap.  With high value (r >> 1), terrain will exhibit 'rough'
        terrain, while a low value << 1, the terrain will be smooth.
    * random_corder (bool):
        Specifies whether grid corner points are initialised at 0 (False) or
        populated with random values (True).  By default grid corner points are
        assigned random values (rand() * scale).

    Returns:
        ndarray grid representing a heightmap.

    """
    if not np.log2(size[0] - 1).is_integer():
        raise ValueError('Grid size {} not allowed, 2**n + 1 accepted where '
            'n in Z'.format(size[0]))
    if size[0] != size[1]:
        raise ValueError(
            'Grid dimensions specified: {} are not equal'.format(size))

    grid = np.ma.zeros(size, dtype=np.float)

    if random_corner:
        corner_x = np.array([0, 0, -1, -1])
        corner_y = np.array([0, -1, -1, 0])
        grid[corner_x, corner_y] = perturbation(scale, 4)

    iterations = (
        np.log2(size[0] - 1).astype(int), np.log2(size[1] - 1).astype(int))

    for iteration in xrange(iterations[0]):
        xsize, ysize, xstep, ystep = subdivide_grid(iteration, size)

        # Diamond step.
        for step_x in xrange(xstep, size[0], (xstep) * 2):
            for step_y in xrange(ystep, size[1], (ystep) * 2):
                center_x = np.array([step_x - xstep,
                                     step_x + xstep,
                                     step_x + xstep,
                                     step_x - xstep])
                center_y = np.array([step_y + ystep,
                                     step_y + ystep,
                                     step_y - ystep,
                                     step_y - ystep])

                grid[step_y, step_x] = (grid[center_y, center_x].mean() +
                                        perturbation(scale))

        # Square step.
        for step_x in xrange(xstep, size[0], (xstep) * 2):
            for step_y in xrange(ystep, size[1], (ystep) * 2):
                center_y, center_x = square_det(
                    xstep, ystep, step_x, step_y, size)

                # Determine value for each of these midpoints.
                for y, x in zip(center_y, center_x):
                    center_in_y, center_in_x = square_det(
                        xstep, ystep, x, y, size, wrap=False)
                    grid[y, x] = (grid[center_in_y, center_in_x].mean() +
                                  perturbation(scale))

        # Update perturbation constant.
        scale *= np.power(2, -1.)

        print 'iteration: {} of {} completed'.format(
            iteration + 1, iterations[0])
        print 'xsize: {}, ysize: {}, xstep: {}, ystep: {}\n'.format(
            xsize, ysize, xstep, ystep)

    return grid


def plot_grid(grid):
    """
    Simple plotting wrapper for plotting our resulting heightmap grid.

    Resulting 3D wireframe automatically determines suitable z-limits based on
    value ranges.

    Args:

    * grid (:class:`numpy.array`):
        A grid representing a heightmap.

    Returns:
        None

    """
    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')
    xx, yy = np.meshgrid(np.arange(grid.shape[0]), np.arange(grid.shape[1]))
    ax.plot_wireframe(xx, yy, grid, color='b')
    diff = grid.max() - grid.min()
    ax.set_zlim3d(grid.min() - (3 * diff), grid.max() + (3 * diff))

    ax2 = fig.add_subplot(122)
    mesh = ax2.pcolormesh(grid)
    plt.colorbar(mesh)
    plt.show()


def maya_plot(grid):
    mlab.figure(size=(800, 600), bgcolor=(0.16, 0.28, 0.46))
    mlab.surf(grid, colormap='gist_earth', vmin=0, warp_scale="auto")
    mlab.show()


PLOT_FUNCTION = {'mayavi':  maya_plot,
                 'matplotlib': plot_grid}


if __name__ == '__main__':
    size = ((257, 257), (65, 65), (33, 33), (17, 17), (5, 5), (3, 3))
    size = size[0]

    np.random.seed(0)

    grid = midpoint_sd(size)
    PLOT_FUNCTION[PLOT_UTIL](grid)
