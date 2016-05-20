#!/usr/bin/env python

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pylab
import matplotlib.pyplot as plt

from sklearn.decomposition import RandomizedPCA
from PIL import Image

def pcaAndPlot(X, x_to_centroids, centroids, no_dims = 2):
    pca = RandomizedPCA(n_components=no_dims)
    x_trans = pca.fit_transform(X)
    x_sizes = np.full((x_trans.shape[0]), 30, dtype=np.int)
    plt.scatter(x_trans[:, 0], x_trans[:, 1], s=x_sizes, c=x_to_centroids)
    centroids_trans = pca.fit_transform(centroids)
    centroids_col = np.arange(centroids.shape[0])
    centroids_sizes = np.full((centroids.shape[0]), 70, dtype=np.int)
    plt.scatter(centroids_trans[:, 0], centroids_trans[:, 1], s=centroids_sizes, c=centroids_col)
    plt.show()

def plotError(train, validation, test, path, name):
    plt.xlabel('Epoch')
    plt.ylabel('Error (%)')
    plt.figure(1)
    lines = plt.plot(train[0], train[1], validation[0], validation[1], \
        test[0], test[1])
    plt.setp(lines[0], color='g', linewidth=2.0, label='Training')
    plt.setp(lines[1], color='b', linewidth=2.0, label='Validation')
    plt.setp(lines[2], color='r', linewidth=2.0, label='Testing')

    g_patch = mpatches.Patch(color='g', label='Training Loss Curve')
    b_patch = mpatches.Patch(color='b', label='Validation Loss Curve')
    r_patch = mpatches.Patch(color='r', label='Testing Loss Curve')

    plt.legend(handles=[g_patch, b_patch, r_patch]) #, bbox_to_anchor=(.95, 0.25))

    plt.title('Error / Epoch')
    plt.grid(True)
    plt.savefig(path + '/' +  name)

def arraysToImgs(rows,colums,arr,path,out_shape=None):
    image = Image.fromarray(tile_raster_images(
        X=arr,
        img_shape=out_shape, tile_shape=(rows, colums),
        tile_spacing=(1, 1)))
    image.save(path)


def arrayToImg(arr, path, out_shape=None):
    if out_shape is not None:
        arr = arr.reshape(out_shape)
    matplotlib.image.imsave(path, arr, cmap='Greys_r')

def scale_to_unit_interval(ndar, eps=1e-8):
    """ Scales all values in the ndarray ndar to be between 0 and 1 """
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar

def tile_raster_images(X, img_shape, tile_shape, tile_spacing=(0, 0),
                         scale_rows_to_unit_interval=True,
                         output_pixel_vals=True):
    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    # The expression below can be re-written in a more C style as
    # follows :
    #
    # out_shape = [0,0]
    # out_shape[0] = (img_shape[0] + tile_spacing[0]) * tile_shape[0] -
    #                tile_spacing[0]
    # out_shape[1] = (img_shape[1] + tile_spacing[1]) * tile_shape[1] -
    #                tile_spacing[1]
    out_shape = [(ishp + tsp) * tshp - tsp for ishp, tshp, tsp
                        in zip(img_shape, tile_shape, tile_spacing)]

    if isinstance(X, tuple):
        assert len(X) == 4
        # Create an output numpy ndarray to store the image
        if output_pixel_vals:
            out_array = np.zeros((out_shape[0], out_shape[1], 4), dtype='uint8')
        else:
            out_array = np.zeros((out_shape[0], out_shape[1], 4), dtype=X.dtype)

        #colors default to 0, alpha defaults to 1 (opaque)
        if output_pixel_vals:
            channel_defaults = [0, 0, 0, 255]
        else:
            channel_defaults = [0., 0., 0., 1.]

        for i in range(4):
            if X[i] is None:
                # if channel is None, fill it with zeros of the correct
                # dtype
                out_array[:, :, i] = np.zeros(out_shape,
                        dtype='uint8' if output_pixel_vals else out_array.dtype
                        ) + channel_defaults[i]
            else:
                # use a recurrent call to compute the channel and store it
                # in the output
                out_array[:, :, i] = tile_raster_images(X[i], img_shape, tile_shape, tile_spacing, scale_rows_to_unit_interval, output_pixel_vals)
        return out_array

    else:
        # if we are dealing with only one channel
        H, W = img_shape
        Hs, Ws = tile_spacing

        # generate a matrix to store the output
        out_array = np.zeros(out_shape, dtype='uint8' if output_pixel_vals else X.dtype)


        for tile_row in range(tile_shape[0]):
            for tile_col in range(tile_shape[1]):
                if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                    if scale_rows_to_unit_interval:
                        # if we should scale values to be between 0 and 1
                        # do this by calling the `scale_to_unit_interval`
                        # function
                        this_img = scale_to_unit_interval(X[tile_row * tile_shape[1] + tile_col].reshape(img_shape))
                    else:
                        this_img = X[tile_row * tile_shape[1] + tile_col].reshape(img_shape)
                    # add the slice to the corresponding position in the
                    # output array
                    out_array[
                        tile_row * (H+Hs): tile_row * (H + Hs) + H,
                        tile_col * (W+Ws): tile_col * (W + Ws) + W
                        ] \
                        = this_img * (255 if output_pixel_vals else 1)
        return out_array