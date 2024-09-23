import matplotlib.pyplot as plt
import numpy as np
from stardist.models import StarDist2D, Config2D
import json
import geojson
from typing import List, Tuple
from pathlib import Path
import os
from tifffile import imread, imwrite
from tqdm import tqdm
import random
from PIL import Image
from stardist import fill_label_holes
import copy
from tensorflow.python.summary.summary_iterator import summary_iterator
import struct
from matplotlib.colors import ListedColormap
import pandas as pd


def load_model(model_path: str) -> StarDist2D:
    """Justin made: Load StarDist model weights, configurations, and thresholds"""
    # TODO: remove offshoot thing
    with open(model_path + '\\config.json', 'r') as f:
        config = json.load(f)
    with open(model_path + '\\thresholds.json', 'r') as f:
        thresh = json.load(f)
    model = StarDist2D(config=Config2D(**config), basedir=model_path, name='offshoot_model')
    model.thresholds = thresh
    print('Overriding defaults:', model.thresholds, '\n')
    model.load_weights(model_path + '\\weights_best.h5')
    return model


def load_published_he_model(folder_to_write_new_model_folder: str, name_for_new_model: str) -> StarDist2D:
    """Justin made: Load HE published model"""
    # TODO: remove offshoot thing
    published_model = StarDist2D.from_pretrained('2D_versatile_he')
    original_thresholds = copy.copy({'prob': published_model.thresholds[0], 'nms': published_model.thresholds[1]})
    configuration = Config2D(n_channel_in=3, grid=(2, 2), use_gpu=True, train_patch_size=[256, 256])
    model = StarDist2D(config=configuration, basedir=folder_to_write_new_model_folder, name=name_for_new_model)
    model.keras_model.set_weights(published_model.keras_model.get_weights())
    model.thresholds = original_thresholds
    return model


def read_tiles(tiles_pth: str) -> List[np.ndarray]:
    """Read .tif tile files from pth and return list of image matrices"""
    tiles_full_pth = [os.path.join(tiles_pth, tile) for tile in os.listdir(tiles_pth) if tile.endswith('.tif')]
    tiles = [imread(tile_pth) for tile_pth in tiles_full_pth]
    return tiles


def read_masks(masks_pth: str) -> List[np.ndarray]:
    """Reads .tif file of mask and returns a np.array list. Fills holes in the annotatinos that were left"""
    masks = read_tiles(masks_pth)
    masks_fixed = [np.array(fill_label_holes(y)) for y in masks]
    return masks_fixed


def segment_tiles(tiles: List[np.ndarray], model: StarDist2D) -> List[np.ndarray]:
    """returns a list of tuples from input tile. The tuple is made up of an image, as well as a dictionary of the
    contour results.
    tiles: input of normalized (/255) list of tiles"""

    y_pred = [model.predict_instances(x, n_tiles=model._guess_n_tiles(x), show_tile_progress=False)[0]
              for x in tqdm(tiles)]

    return y_pred


def show_HE_and_segmented(HE_im: np.ndarray, segmented: np.ndarray, **kwargs) -> None:
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


def segment_dir_of_images(WSI_path: str, file_type: str, out_nm: str, model: StarDist2D, save_tif: bool):
    """Segments a directory of WSIs
    WSI_path: path to H&E images that will get their nuclei segmented
    save_tif: option to save tif image of segmentation"""

    # make list of full paths to images
    WSIs = [os.path.join(WSI_path, f) for f in os.listdir(WSI_path) if f.endswith(file_type)]

    # make outdirs
    out_pth = os.path.join(WSI_path, out_nm)
    if not os.path.exists(out_pth):
        os.mkdir(out_pth)

    out_pth_json = os.path.join(out_pth, 'json')
    out_pth_tif = os.path.join(out_pth, 'tif')

    if not os.path.exists(out_pth_json):
        os.mkdir(out_pth_json)
    if not os.path.exists(out_pth_json):
        os.mkdir(out_pth_json)
    if not os.path.exists(out_pth_tif) and save_tif:
        os.mkdir(out_pth_tif)

    # main loop
    for img_pth in WSIs:
        try:
            name = os.path.basename(img_pth)

            if not os.path.exists(os.path.join(out_pth_json, (name[:-5] + '.json'))):
                print(f'Starting {name}')
                
                img = imread(img_pth)
                img = img/255  # normalization used to train model
                
                if not save_tif:
                    _, polys = model.predict_instances_big(img, axes='YXC', block_size=4096, min_overlap=128, context=128, n_tiles=(4,4,1))
                    # labels, polys = model.predict_instances_big(img, axes='YXC', block_size=4096, min_overlap=128, context=128, n_tiles=(4,4,1))
        
                    print('Saving json...')
                    save_json_from_WSI_pred(polys, out_pth_json, name)

                else:
                    # tif file is like 3 GB usually, so only uncomment next part if you are ok with that
                    labels, polys = model.predict_instances_big(img, axes='YXC', block_size=4096, min_overlap=128, context=128, n_tiles=(4,4,1))
                    print('Saving json...')
                    save_json_from_WSI_pred(polys, out_pth_json, name)
                    
                    print('Saving tif...')
                    imwrite(os.path.join(out_pth_tif, name[:-5] + '.tif'), labels)
                    
            else:
                print(f'Skipping {name}')
        except:
            print(f'skipping {img_pth}, probably bc its too big...')


def augment_tiles(tiles: List[np.ndarray], masks: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Augments the intput HE tiles/mask tiles pair by flipping and/or rotating images. Adds 7 augmented images per
    original image in the input set.
    tiles: list of HE images
    masks: list of masks of nuclei segmentations"""

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

    HE_aug = [np.array(i) for x in HE_aug for i in x]  # flatten the list of lists
    mask_aug = [np.array(i) for x in mask_aug for i in x]  # flatten the list of lists

    return HE_aug, mask_aug


def split_train_val_set(tiles: np.ndarray, masks: np.ndarray, val_ratio: float):
    """Splits the input set of HE image/mask pairs into training and validation sets.
    tiles: list of HE images
    masks: list of masks of nuclei segmentations
    val_ratio: value between 0 and 1. The ratio of tiles used for validation as opposed to training."""

    assert len(tiles) == len(masks)

    n_tiles = len(tiles)
    num_indices = round(n_tiles * val_ratio)

    val_indices = sorted(random.sample(range(n_tiles), num_indices))
    train_indices = sorted(list(set(range(n_tiles)) - set(val_indices)))

    tiles_train = [tiles[i] for i in train_indices]
    masks_train = [masks[i] for i in train_indices]

    tiles_val = [tiles[i] for i in val_indices]
    masks_val = [masks[i] for i in val_indices]

    return tiles_train, masks_train, tiles_val, masks_val


def normalize_images(tiles: List[np.ndarray]) -> List[np.ndarray]:
    """Normalizes H&E images by dividing by 255."""
    tiles_norm = [np.divide(tile, 255) for tile in tiles]
    return tiles_norm


def get_loss_data(pth_training_log, pth_out) -> list:
    """Saves a .txt file with loss data for each epoch of training"""

    # event_file = r"\\10.99.68.178\andreex\data\Stardist\models\monkey_ft_11_02_2023_lr_1e-4_epochs_200_pt_10\logs\train\events.out.tfevents.1698964029.WPC-C13.20400.7.v2"

    loss_values = []

    for summary in summary_iterator(pth_training_log):
        for value in summary.summary.value:
            if value.tag == 'epoch_loss':
                loss = struct.unpack('f', value.tensor.tensor_content)[0]
                loss_values.append(loss)

    out_txt_name = f"{pth_out}\loss.txt"

    with open(out_txt_name, 'w') as f:
        f.write('\n'.join(map(str, loss_values)) + '\n')

    return loss_values


def plot_predictions_vs_gt(tile: np.ndarray, gt_mask: np.ndarray, pred_mask: np.ndarray, cmap: ListedColormap) -> None:
    """Visualizes predictions and ground truth for a given tile."""
    fig, ax = plt.subplots(1, 3, figsize=(24, 8))
    ax[0].imshow(tile)
    ax[0].axis('off')
    ax[0].set_title('H&E')

    ax[1].imshow(tile)
    ax[1].imshow(gt_mask, cmap=cmap, alpha=0.5)
    ax[1].axis('off')
    ax[1].set_title('Ground Truth')

    ax[2].imshow(tile)
    ax[2].imshow(pred_mask, cmap=cmap, alpha=0.5)
    ax[2].axis('off')
    ax[2].set_title('Predicted')


class TileSetScorer:
    """
    Assumes the vast majority of objects to have internal centroids (i.e. convex)
    """
    def __init__(self, base_names: list[str], gt_set: list[np.ndarray],
                 pred_set: list[np.ndarray], taus: list[float]):
        self.base_names = base_names
        self.gt_set = gt_set
        self.pred_set = pred_set
        self.taus = taus
        self.df_results_granular = self.score_set()
        self.df_results_summary = self.summarize_scores(self.df_results_granular)

    # @staticmethod
    def score_set(self) -> pd.DataFrame:
        # Initialize an empty dataframe to store results
        columns = ['Image', 'Tau', 'IoU', 'TP', 'FP', 'FN',
                        'Precision', 'Recall', 'Avg Precision', 'F1 Score', 'Seg Quality', 'Pan Quality']
        df_results = pd.DataFrame(columns=columns)
        for i, base_name in enumerate(self.base_names):
            gt, pred = self.gt_set[i], self.pred_set[i]
            for tau in self.taus:
                # Make one line dataframe to concatenate to results
                results = {'Image': [base_name], 'Tau': [tau]}
                offset = len(results)
                scores = ScoringSubroutine(gt, pred, tau).scores
                for j, score in enumerate(scores):
                    results[columns[j + offset]] = score
                df_results = pd.concat([df_results, pd.DataFrame(results)], axis=0, ignore_index=True)
        return df_results

    @staticmethod
    def summarize_scores(df_granular) -> pd.DataFrame:
        df_summary = \
            df_granular.groupby(['Image']).agg({'IoU': 'median', 'Avg Precision': 'mean'}).reset_index()
        df_summary.columns = ['Image', 'IoU', 'mAP']
        return df_summary


class ScoringSubroutine:
    """
    Assumes the vast majority of objects to have internal centroids (i.e. convex)
    """
    def __init__(self, gt: np.ndarray, pred: np.ndarray, tau: float):
        gt_centroids = self.find_centroids(gt)
        pred_centroids = self.find_centroids(pred)
        self.scores = self.calculate_scores(gt, pred, tau, gt_centroids, pred_centroids)

    @staticmethod
    def find_centroids(mask: np.ndarray) -> list[list[int, int]]:
        # Finds centroid coordinates as weighted averages of binary pixel values
        centroids = []
        for object_id in np.unique(mask)[1:]:
            binary_mask = (mask == object_id)
            x_coords, y_coords = np.where(binary_mask)
            x, y = int(np.round(np.mean(x_coords))), int(np.round(np.mean(y_coords)))
            centroids.append([x, y])
        return centroids

    def calculate_scores(self, gt: np.ndarray, pred: np.ndarray, tau: float,
                         gt_centroids: list[list[int, int]], pred_centroids: list[list[int, int]]) \
            -> (float, int, int, int, float, float, float, float, float, float):
        iou = self.calc_iou(gt, pred)
        tp, fp, seg_qual = self.calc_tp_fp_sg(gt, pred, tau, pred_centroids)
        fn = self.calc_fn(gt, pred, tau, gt_centroids)
        if not tp:
            precision, recall, avg_precision, f1 = 0, 0, 0, 0
        else:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            avg_precision = tp / (tp + fp + fn)
            f1 = 2 * precision * recall / (precision + recall)
        pan_qual = seg_qual * f1
        return iou, tp, fp, fn, precision, recall, avg_precision, f1, seg_qual, pan_qual

    @staticmethod
    def calc_iou(array1: np.ndarray or bool, array2: np.ndarray or bool) -> float:
        # Compares pixel-to-pixel coverage of any pixel greater than 0
        intersection = np.logical_and(array1, array2)
        union = np.logical_or(array1, array2)
        intersection_area = np.sum(intersection)
        union_area = np.sum(union)
        return intersection_area / union_area

    def calc_tp_fp_sg(self, gt: np.ndarray, pred: np.ndarray, tau: float, pred_centroids: list[list[int, int]]) \
            -> (int, int, float):
        # Assumes the vast majority of object centroids are internal (i.e. convex objects)
        tp, fp, sum_tp_iou = 0, 0, 0.0
        for centroid in pred_centroids:
            x, y = centroid[0], centroid[1]
            gt_val_at_pred_centroid = gt[x][y]
            pred_val_at_pred_centroid = pred[x][y]
            if gt_val_at_pred_centroid:
                binary_mask_gt = (gt == gt_val_at_pred_centroid)
                binary_mask_pred = (pred == pred_val_at_pred_centroid)
                iou = self.calc_iou(binary_mask_gt, binary_mask_pred)
                if iou >= tau:
                    tp += 1
                    sum_tp_iou += iou
                else:
                    fp += 1
            else:
                fp += 1
        sg = sum_tp_iou / tp if tp > 0 else 0
        return tp, fp, sg

    def calc_fn(self, gt: np.ndarray, pred: np.ndarray, tau: float, gt_centroids: list[list[int, int]]) -> int:
        fn = 0
        for centroid in gt_centroids:
            x, y = centroid[0], centroid[1]
            pred_val_at_gt_centroid = pred[x][y]
            gt_val_at_gt_centroid = gt[x][y]
            if pred_val_at_gt_centroid:
                binary_mask_gt = (gt == gt_val_at_gt_centroid)
                binary_mask_pred = (pred == pred_val_at_gt_centroid)
                iou = self.calc_iou(binary_mask_gt, binary_mask_pred)
                if iou < tau:
                    fn += 1
            else:
                fn += 1
        return fn


def get_stats(HE_tiles_pth: str, mask_gt_tiles: List[np.ndarray], mask_pred_tiles: List[np.ndarray], taus: list) -> pd.DataFrame:
    """Return a df with stats about each tile."""
    nms = [os.path.basename(file) for file in os.listdir(HE_tiles_pth) if file.endswith('.tif')]
    scores = TileSetScorer(nms, mask_gt_tiles, mask_pred_tiles, taus)
    results = scores.score_set()
    return results


def make_f1_plot(HE_tiles_pth: str, results: pd.DataFrame, taus: list) \
        -> None:
    """idk yet"""
    nms = [os.path.basename(file) for file in os.listdir(HE_tiles_pth) if file.endswith('.tif')]
    names = results['Image']

    names = [name.split(".")[0][21:] for name in names]  # this should be a list of the numbers at end of file names

    for i in range(len(names)):
        if len(names[i]) > 6:
            names[i] = names[i][:5]

    f1_scores = results['F1 Score']

    index = np.arange(len(nms))

    # Plotting the bars
    fig = plt.figure(figsize=(25, 10))
    fig.set_facecolor('white')

    plt.bar(index, f1_scores, color='darksalmon')

    plt.xlabel("Tile Name", fontsize=20)
    plt.ylabel("F1 Score", fontsize=20)
    plt.title("F1 Scores in Testing Tiles (tau = 0.7)", fontsize=28)
    plt.axhline(y=0.7, linestyle='--', color='red', label=f'Target F1 = {taus[0]}')
    plt.ylim(0, 1)
    plt.xticks(index, names)  # Set x-axis labels to tile names
    plt.legend(fontsize=20)
    plt.show()
    return


def save_json_from_WSI_pred(result, out_pth, name):
    """Saves a json file with centroids and contours for StarDist output."""
    coords = result['coord']
    points = result['points']

    json_data = []

    for i in range(len(points)):
        point = points[i]
        contour = coords[i]
        centroid = [int(point[0]), int(point[1])]  # TODO: FIX
        contour = [[float(coord) for coord in xy[::-1]] for xy in contour]

        # Create a new dictionary for each contour
        dict_data = {
            "centroid": [centroid],
            "contour": [contour]
        }

        json_data.append(dict_data)

    new_fn = name[:-5] + '.json'

    with open(os.path.join(out_pth, new_fn),'w') as outfile:
        json.dump(json_data, outfile)
    print('Finished',new_fn)


def format_seg_data(segmentation_data, ds):
    data_list = []
    for data in segmentation_data:
        centroid = data['centroid'][0]
        contour = data['contour'][0]

        ds_centroid = [int(c / ds) for c in centroid]
        ds_contour = [[value / ds for value in sublist] for sublist in contour]
        ds_contour = [[round(x, 2), round(y, 2)] for x, y in zip(ds_contour[0], ds_contour[1])]

        dat = [ds_centroid, ds_contour]
        data_list.append(dat)
    return data_list











