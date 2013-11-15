import numpy as np
import numpy.lib.stride_tricks as stride
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt


#size = (257, 257)
#size = (65, 65)
#size = (33, 33)
size = (17, 17)
#size = (5, 5)
#size = (3, 3)

dat = np.ma.zeros(size, dtype=np.float)
#dat.mask = True

np.random.seed(0)
corners = np.array([[0, 0], [0, -1], [-1, -1], [-1, 0]])
corner_x = np.array([0, 0, -1, -1])
corner_y = np.array([0, -1, -1, 0])
dat[corner_x, corner_y] = np.random.rand(4) * 1

iterations = (np.log2(size[0] - 1).astype(int), np.log2(size[1] - 1).astype(int))


def window_views(xsize=3, ysize=3, xstep=1, ystep=1):
    """
    Generate view ndarray of the grid.

    Args:

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


    """
    strides = (dat.strides[0] * xstep,
               dat.strides[1] * ystep,
               dat.strides[0],
               dat.strides[1])
    window = (((dat.shape[0] - 1) / (xsize - 1)),
              ((dat.shape[1] - 1) / (ysize - 1)),
              xsize,
              ysize)
    all_windows = stride.as_strided(dat, window, strides)
    return all_windows


def subdivide_grid(iteration):
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

    # Double checking to ensrure compliance with algorithm.
    if xstep != ystep:
        raise RuntimeError('step sizes do not match')
    if xsize != ysize:
        raise RuntimeError('window dimensions not equal')

    return xsize, ysize, xstep, ystep


def perturbation(constant):
    return np.random.rand() * 2 * constant - constant


def square_det(xstep, ystep, step_x, step_y, wrap=True):
    # Return index of all edge midpoints of surrounding point.
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
    

def main(scale=0.5):
    for iteration in xrange(iterations[0]):
        xsize, ysize, xstep, ystep = subdivide_grid(iteration)

        # Diamond step.
        for step_x in xrange(xstep, size[0], (xstep)*2):
            for step_y in xrange(ystep, size[1], (ystep)*2):
                # Each operation has -1 as array index from 0.  Cancels out
                # with subtraction but through with addition.
                center_x = np.array([step_x - xstep,
                                     step_x + xstep,
                                     step_x + xstep,
                                     step_x - xstep])
                center_y = np.array([step_y + ystep,
                                     step_y + ystep,
                                     step_y - ystep,
                                     step_y - ystep])

                dat[step_x, step_y] = (dat[center_y, center_x].mean() +
                                       perturbation(scale))

        # Square step.
        for step_x in xrange(xstep, size[0], (xstep)*2):
            for step_y in xrange(ystep, size[1], (ystep)*2):
                center_y, center_x = square_det(xstep, ystep, step_x, step_y)

                # Determine value for each of these midpoints.
                for y, x in zip(center_y, center_x):
                    # Add one to indexes as I counter indexes in square_det
                    center_in_y, center_in_x = square_det(
                        xstep, ystep, x, y, wrap=False)
                    dat[y, x] = (dat[center_in_y, center_in_x].mean() +
                                 perturbation(scale))

        # Update perturbation constant.
        pert_cons /= 2.#*= np.power(2, -1.)

        #print dat[xstep - 1:: xstep, ystep - 1:: ystep]
        print 'iteration: {} of {} completed'.format(
            iteration + 1, iterations[0])
        print 'xsize: {}, ysize: {}, xstep: {}, ystep: {}\n'.format(
            xsize, ysize, xstep, ystep)


def plot_grid():
    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')
    #ax.set_zlim3d(-10, 10)
    xx, yy = np.meshgrid(np.arange(size[0]), np.arange(size[1]))
    ax.plot_wireframe(xx, yy, dat, color='b')
    ax.set_zlim3d(-2, 2)

    ax2 = fig.add_subplot(122)
    mesh = ax2.pcolormesh(dat)
    plt.colorbar(mesh)


if __name__ == '__main__':
    main()
    plot_grid()
    plt.show()
