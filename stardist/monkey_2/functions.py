import matplotlib.pyplot as plt
import numpy as np
from stardist.models import StarDist2D, Config2D
import json
import geojson
from typing import List, Tuple
from pathlib import Path
import os
from tifffile import imread
from tqdm import tqdm
import random
from PIL import Image


def load_model(model_path: str) -> StarDist2D:
    """Load StarDist model weights, configurations, and thresholds"""
    with open(model_path + '\\config.json', 'r') as f:
        config = json.load(f)
    with open(model_path + '\\thresholds.json', 'r') as f:
        thresh = json.load(f)
    model = StarDist2D(config=Config2D(**config), basedir=model_path, name='offshoot_model')
    model.thresholds = thresh
    print('Overriding defaults:', model.thresholds, '\n')
    model.load_weights(model_path + '\\weights_best.h5')
    return model


def read_tiles(tiles_pth: str) -> List[np.array]:
    """Read .tif tile files from pth and return list of image matrices"""
    tiles_full_pth = [os.path.join(tiles_pth, tile) for tile in os.listdir(tiles_pth) if tile.endswith('.tif')]
    tiles = [imread(tile_pth) for tile_pth in tiles_full_pth]
    return tiles


def segment_tiles(tiles: List[np.array], model: StarDist2D) -> List[np.array]:
    """returns a list of tuples from input tile. The tuple is made up of an image, as well as a dictionary of the
    contour results."""

    tiles_norm = [tile / 255 for tile in tiles]  # divide by 255 to get into (0,1) range

    y_pred = [model.predict_instances(x, n_tiles=model._guess_n_tiles(x), show_tile_progress=False)[0]
              for x in tqdm(tiles_norm)]

    return y_pred


def show_HE_and_segmented(HE_im: np.array, segmented: np.array, **kwargs) -> None:
    """Show H&E image on left, Segmentation on right."""

    if HE_im.shape[0:2] == segmented.shape[0:2]:
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))

        # Plot the original image on the left
        ax[0].imshow(HE_im, **kwargs)

        # Plot the segmented image on the right
        ax[1].imshow(segmented, **kwargs)

        ax[0].axis('off')
        ax[1].axis('off')

        plt.tight_layout()
        plt.show()
    else:
        print("H&E image is not same shape as segmented image.")


def save_geojson_from_segmentation(tiles_pth: str, model: StarDist2D, outpth: str) -> None:
    """Save a geojson file from stardist segmentation in order to load the results into QuPath.
    tiles_pth: path to the folder in which the .tif files of each tile are found
    model: the StarDist model, get by running load_model
    outpth: location where to save"""
    outpth = Path(outpth)

    tiles = [_ for _ in os.listdir(tiles_pth) if _.endswith('tif')]

    for cc in range(len(tiles)):
        name = tiles[cc]
        tile_pth = os.path.join(tiles_pth, name)
        tile = imread(tile_pth)

        tile = tile / 255

        result = model.predict_instances(tile)

        # save centroids and contours in geojson format to import into qupath
        coords = result[1]['coord']
        # print(len(coords[0][0]))
        contours = []
        for xy in coords:
            contour = []
            for i in range(len(xy[0])):
                p = [xy[0][i], xy[1][i]]  # [x, y]
                contour.append(p)
            contours.append(contour)

        data_stardist = []
        for i in range(len(result[1]['points'])):
            nucleus = result[1]['points'][i]
            contour = contours[i]
            both = [nucleus, contour]
            data_stardist.append(both)

        GEOdata = []

        for centroid, contour in data_stardist:
            centroid = [centroid[0] + 0, centroid[1] + 0]
            # xy coordinates are swapped, so I reverse them here with xy[::-1]
            # note: add 1 to coords to fix 0 indexing vs 1 index offset
            contour = [[coord + 0 for coord in xy[::-1]] for xy in contour]  # Convert coordinates to integers
            contour.append(contour[0])  # stardist doesn't close the circle, needed for qupath

            # Create a new dictionary for each contour
            dict_data = {
                "type": "Feature",
                "id": "PathCellObject",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [contour]
                },
                "properties": {
                    'objectType': 'annotation',
                    'classification': {'name': 'Nuclei', 'color': [97, 214, 59]}
                }
            }

            GEOdata.append(dict_data)

        new_fn = name[:-4] + '.geojson'

        with open(outpth.joinpath(new_fn), 'w') as outfile:
            geojson.dump(GEOdata, outfile)
        print('Finished', new_fn)


def _get_train_val_split(n_tiles: int, val_ratio: float) -> Tuple[list, list]:
    """Returns a list of indices for training and validation for the given tiles.
    n_tiles: number of tiles in training/validation set
    val_ratio: value between 0 and 1. The ratio of tiles used for validation as opposed to training."""

    num_indices = round(n_tiles * val_ratio)

    val_indices = sorted(random.sample(range(n_tiles), num_indices))
    train_indices = sorted(list(set(range(n_tiles)) - set(val_indices)))

    return train_indices, val_indices


def _split_augmented_data(HE_aug_norm_list, mask_aug_norm_list):
    """
    Splits augmented HE images and mask images into training and validation sets.

    Args:
      HE_aug_norm: A list of NumPy arrays representing the normalized augmented HE images.
      mask_aug_norm: A list of NumPy arrays representing the normalized augmented mask images.

    Returns:
      X_trn, Y_trn, X_val, Y_val: Training and validation sets of HE images and mask images.
    """

    # Create empty lists to store the training and validation sets.
    X_trn = []
    Y_trn = []
    X_val = []
    Y_val = []

    for i in range(len(HE_aug_norm_list)):
        HE_aug_norm = HE_aug_norm_list[i]
        mask_aug_norm = mask_aug_norm_list[i]

        # Assert that there is enough training data.
        assert len(HE_aug_norm) > 1, "not enough training data"

        # Create a random number generator with a fixed seed for reproducibility.
        rng = np.random.RandomState(42)

        # Permute the indices of the augmented data.
        ind = rng.permutation(len(HE_aug_norm))

        # Split the indices into training and validation indices.
        n_val = max(1, int(round(0.15 * len(ind))))
        ind_trn, ind_val = ind[:-n_val], ind[-n_val:]

        # Split the augmented data into training and validation sets.
        for kk in ind_trn:
            X_trn.append(HE_aug_norm[kk])
            Y_trn.append(mask_aug_norm[kk])

        for lol in ind_val:
            X_val.append(HE_aug_norm[lol])
            Y_val.append(mask_aug_norm[lol])

    print(f'number of images: {len(HE_aug_norm_list * len(HE_aug_norm_list[0]))}')
    print(f'- training: {len(X_trn)}')
    print(f'- validation: {len(X_val)}')

    X_trn = [np.array(x) for x in X_trn]
    Y_trn = [np.array(y) for y in Y_trn]

    X_val = [np.array(x) for x in X_val]
    Y_val = [np.array(y) for y in Y_val]

    # Return the training and validation sets.
    return X_trn, Y_trn, X_val, Y_val


def augment_tiles(tiles: List[np.ndarray], masks: List[np.ndarray], val_ratio: float) -> Tuple[List[List], List[List]]:
    """idk yet
    val_ratio: value between 0 and 1. The ratio of tiles used for validation as opposed to training."""

    assert len(tiles) == len(masks)

    HE_aug = [[] for _ in range(len(tiles))]
    mask_aug = [[] for _ in range(len(masks))]

    for i in range(len(tiles)):
        im = Image.fromarray(tiles[i])
        lbl = Image.fromarray(masks[i])

        # add original unaltered images
        HE_aug[i].append(im)
        mask_aug[i].append(lbl)

        # Rotate the image and label 90 degrees three times.
        for _ in range(3):
            im = im.rotate(90)
            HE_aug[i].append(im)
            lbl = lbl.rotate(90)
            mask_aug[i].append(lbl)

        # Flip the image and label horizontally.
        im = Image.fromarray(tiles[i])
        flipped_im = im.transpose(Image.FLIP_LEFT_RIGHT)

        lbl = Image.fromarray(masks[i])
        flipped_lbl = lbl.transpose(Image.FLIP_LEFT_RIGHT)

        HE_aug[i].append(flipped_im)
        mask_aug[i].append(flipped_lbl)

        # Rotate the flipped image and label 90 degrees three times.
        for _ in range(3):
            flipped_im = flipped_im.rotate(90)
            HE_aug[i].append(flipped_im)
            flipped_lbl = flipped_lbl.rotate(90)
            mask_aug[i].append(flipped_lbl)

    train_indices, val_indices = _get_train_val_split(n_tiles=len(tiles), val_ratio=val_ratio)

    return HE_aug, mask_aug










