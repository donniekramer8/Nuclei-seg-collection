import numpy as np
import matplotlib.pyplot as plt
from tifffile import imread
from csbdeep.utils.tf import keras_import
from stardist import random_label_cmap
import os
from glob import glob
import pandas as pd
from stardist.models import StarDist2D
from scipy.io import loadmat
import h5py
import cv2


keras = keras_import()


def show_image(img, crop_x, crop_y, tile_size, **kwargs):
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    ax[0].imshow(img, **kwargs)
    cropped_img = img[crop_y:crop_y+tile_size, crop_x:crop_x+tile_size]
    ax[1].imshow(cropped_img, **kwargs)
    ax[0].axis('off')
    ax[1].axis('off')
    plt.tight_layout()
    plt.show()


def load_var_from_mat(mat_file_name, varname):
    try:
        data = loadmat(mat_file_name)
        var = data[varname]
    except:
        data = h5py.File(mat_file_name, 'r')
        var = data[varname][()]
    return var


def format_xy(xy_coords):
    adj_x = [x for x in xy_coords[:, 0]]
    adj_y = [y for y in xy_coords[:, 1]]
    adj_coords = np.transpose(np.array([adj_x, adj_y]))
    return adj_coords


def is_point_encompassed(xy_point, set_of_points):
    x, y = xy_point
    min_x, max_x = min(point[0] for point in set_of_points), max(point[0] for point in set_of_points)
    min_y, max_y = min(point[1] for point in set_of_points), max(point[1] for point in set_of_points)
    return min_x <= x <= max_x and min_y <= y <= max_y


def get_indices_shapes(polys, xy_coords):
    poly_inds = []
    xys = [] # return centroids list

    in_tile = []
    for num, i in enumerate(polys['coord']):
        xy = np.transpose(i)
        # get centroids of poly
        x_cent = np.mean(xy[:, 1])  # reversed order, idk why
        y_cent = np.mean(xy[:, 0])

        # fix orientation of coordinates in matrix
        x = [x for x in xy[:, 0]]
        y = [y for y in xy[:, 1]]
        xy = np.transpose(np.array([x, y]))[:, ::-1]

        in_tile.append([xy])  # just for testing, delete later probably
        # print(xy)

        # instead of searching through entire list of positive cells, just do the ones that are within 100 pixel box
        condition = (
                (xy_coords[:, 0] > (x_cent - 100)) &
                (xy_coords[:, 0] < (x_cent + 100)) &
                (xy_coords[:, 1] > (y_cent - 100)) &
                (xy_coords[:, 1] < (y_cent + 100))
        )
        subset_cords = xy_coords[condition]

        for xy_cy5 in subset_cords:

            if is_point_encompassed(xy_cy5, xy):

                poly_inds.append(num)
                xys.append(xy_cy5)
                # print(num, x_cent, y_cent)
                break  # don't include double counts

    return poly_inds, xys


def get_valid_shapes_for_crop(shapes, crop_x, crop_y, tile_size):
    new_shapes = []
    new_shapes_areas = []
    for i in range(len(shapes)):
        #adj_x = [x - crop_x for x in shapes[i][0, :]]
        #adj_y = [y - crop_y for y in shapes[i][1, :]]
        adj_x = [x - crop_y for x in shapes[i][0, :]]
        adj_y = [y - crop_x for y in shapes[i][1, :]]
        new_shape = np.transpose(np.array([adj_x, adj_y]))
        if all(tile_size > value > 0 for value in adj_x) and all(tile_size > value > 0 for value in adj_y):
            new_shapes.append(new_shape)
            area = cntarea(new_shape)
            new_shapes_areas.append(area)
    return new_shapes, new_shapes_areas


def cntarea(cnt):
    cnt = np.array(cnt).astype(np.float32)
    area = cv2.contourArea(cnt)
    return area


def plot_new_shapes(cropped_img, new_shapes):
    _, ax = plt.subplots(1, 1, figsize=(24, 8))

    ax.imshow(cropped_img)
    ax.axis('off')
    # ax.scatter(adj_Cy5_coords[:,0], adj_Cy5_coords[:,1], c="r")
    ax.set_title('Overlay')

    for polygon in new_shapes:
        x_coords, y_coords = polygon[:, 1], polygon[:, 0]
        # print(x_coords, y_coords)
        x_coords = list(x_coords) + [x_coords[0]]  # Close the polygon
        y_coords = list(y_coords) + [y_coords[0]]  # Close the polygon

        color = 'red'

        ax.plot(x_coords, y_coords, alpha=0.4, color=color)
        ax.fill(x_coords, y_coords, alpha=0.4, color=color)  # Fill the polygon

    # Set labels and title for the plot
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_title('StarDist Segmentation')

    plt.show()


def get_output_df(shapes):
    shapes_areas = []
    for i in range(len(shapes)):
        adj_x = [x for x in shapes[i][0, :]]
        adj_y = [y for y in shapes[i][1, :]]
        new_shape = np.transpose(np.array([adj_x, adj_y]))
        shapes_areas.append(cntarea(new_shape))

    # x, y, area
    output = []

    for i, (x_list, y_list) in enumerate(shapes):
        x = round(np.mean(x_list))
        y = round(np.mean(y_list))
        area = shapes_areas[i]

        row = [x, y, area]
        output.append(row)

    output = np.array(output)
    output = pd.DataFrame(output)
    return output
