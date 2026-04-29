import copy
import json
import os
import random
import struct
from pathlib import Path
from typing import List, Tuple

import geojson
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib.colors import ListedColormap
from PIL import Image
from stardist import fill_label_holes
from stardist.models import Config2D, StarDist2D
from tensorflow.python.summary.summary_iterator import summary_iterator
from tifffile import imread, imwrite
from tqdm import tqdm


def load_model(model_path: str) -> StarDist2D:
    """Load a custom-trained StarDist model from a directory.

    The directory must contain config.json, thresholds.json, and weights_best.h5.
    """
    with open(os.path.join(model_path, 'config.json'), 'r') as f:
        config = json.load(f)
    with open(os.path.join(model_path, 'thresholds.json'), 'r') as f:
        thresh = json.load(f)
    model = StarDist2D(config=Config2D(**config), basedir=model_path, name='offshoot_model')
    model.thresholds = thresh
    print('Overriding defaults:', model.thresholds, '\n')
    model.load_weights(os.path.join(model_path, 'weights_best.h5'))
    return model


def load_published_he_model(folder_to_write_new_model_folder: str, name_for_new_model: str) -> StarDist2D:
    """Load StarDist's pretrained H&E versatile model and re-wrap it for fine-tuning."""
    published_model = StarDist2D.from_pretrained('2D_versatile_he')
    original_thresholds = copy.copy({'prob': published_model.thresholds[0], 'nms': published_model.thresholds[1]})
    configuration = Config2D(n_channel_in=3, grid=(2, 2), use_gpu=True, train_patch_size=[256, 256])
    model = StarDist2D(config=configuration, basedir=folder_to_write_new_model_folder, name=name_for_new_model)
    model.keras_model.set_weights(published_model.keras_model.get_weights())
    model.thresholds = original_thresholds
    return model


def read_tiles(tiles_pth: str) -> List[np.ndarray]:
    """Read all .tif tiles from a directory and return them as a list of arrays."""
    tiles_full_pth = [os.path.join(tiles_pth, tile) for tile in os.listdir(tiles_pth) if tile.endswith('.tif')]
    tiles = [imread(tile_pth) for tile_pth in tiles_full_pth]
    return tiles


def read_masks(masks_pth: str) -> List[np.ndarray]:
    """Read ground-truth mask .tif files and fill any annotation holes."""
    masks = read_tiles(masks_pth)
    masks_fixed = [np.array(fill_label_holes(y)) for y in masks]
    return masks_fixed


def segment_tiles(tiles: List[np.ndarray], model: StarDist2D) -> List[np.ndarray]:
    """Run StarDist prediction on a list of normalized (divided by 255) tiles.

    Returns a list of label arrays (one per tile).
    """
    y_pred = [
        model.predict_instances(x, n_tiles=model._guess_n_tiles(x), show_tile_progress=False)[0]
        for x in tqdm(tiles)
    ]
    return y_pred


def show_HE_and_segmented(HE_im: np.ndarray, segmented: np.ndarray, **kwargs) -> None:
    """Display an H&E image and its segmentation mask side by side."""
    if HE_im.shape[0:2] != segmented.shape[0:2]:
        print("H&E image is not same shape as segmented image.")
        return
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
    ax[0].imshow(HE_im, **kwargs)
    ax[1].imshow(segmented, **kwargs)
    ax[0].axis('off')
    ax[1].axis('off')
    plt.tight_layout()
    plt.show()


def save_geojson_from_segmentation(tiles_pth: str, model: StarDist2D, outpth: str) -> None:
    """Segment tiles and save results as GeoJSON files for import into QuPath.

    Args:
        tiles_pth: Directory containing .tif tile files.
        model: Loaded StarDist model.
        outpth: Directory where .geojson files will be written.
    """
    outpth = Path(outpth)
    tiles = [_ for _ in os.listdir(tiles_pth) if _.endswith('tif')]

    for name in tiles:
        tile_pth = os.path.join(tiles_pth, name)
        tile = imread(tile_pth) / 255
        result = model.predict_instances(tile)

        coords = result[1]['coord']
        contours = []
        for xy in coords:
            contour = [[xy[0][i], xy[1][i]] for i in range(len(xy[0]))]
            contours.append(contour)

        GEOdata = []
        for i, nucleus in enumerate(result[1]['points']):
            centroid = [nucleus[0], nucleus[1]]
            contour = [[coord for coord in xy[::-1]] for xy in contours[i]]
            contour.append(contour[0])
            GEOdata.append({
                "type": "Feature",
                "id": "PathCellObject",
                "geometry": {"type": "Polygon", "coordinates": [contour]},
                "properties": {
                    'objectType': 'annotation',
                    'classification': {'name': 'Nuclei', 'color': [97, 214, 59]}
                }
            })

        new_fn = name[:-4] + '.geojson'
        with open(outpth / new_fn, 'w') as outfile:
            geojson.dump(GEOdata, outfile)
        print('Finished', new_fn)


def json_to_geojson_whole_folder(json_pth: str) -> None:
    """Convert a folder of custom StarDist JSON outputs to GeoJSON for QuPath.

    Skips files that already have a corresponding .geojson in json_pth/geojson/.
    """
    json_pth_list = sorted([
        os.path.join(json_pth, file)
        for file in os.listdir(json_pth)
        if file.endswith(".json")
    ])

    outpth = os.path.join(json_pth, 'geojson')
    os.makedirs(outpth, exist_ok=True)

    for pth in json_pth_list:
        name = os.path.basename(pth)
        new_fn = name[:-4] + 'geojson'
        out_full = os.path.join(outpth, new_fn)

        if os.path.exists(out_full):
            print(f'Skipping {name}... (Already exists)')
            continue

        with open(pth, 'r') as f:
            json_data = json.load(f)

        GEOdata = []
        for nuc in json_data:
            contour = nuc['contour'][0][::-1]
            contour = [[row[i] for row in contour] for i in range(len(contour[0]))]
            contour.append(contour[0])
            GEOdata.append({
                "type": "Feature",
                "id": "PathCellObject",
                "geometry": {"type": "Polygon", "coordinates": [contour]},
                "properties": {
                    'objectType': 'annotation',
                    'classification': {'name': 'Nuclei', 'color': [97, 214, 59]}
                }
            })

        with open(out_full, 'w') as outfile:
            geojson.dump(GEOdata, outfile)
        print('Finished', new_fn)


def segment_dir_of_images(WSI_path: str, file_type: str, out_nm: str, model: StarDist2D, save_tif: bool) -> None:
    """Segment all WSIs in a directory and save results as JSON (and optionally TIF).

    Skips images that already have a corresponding JSON output. Uses
    predict_instances_big with 4096px blocks and 128px overlap for large images.

    Args:
        WSI_path: Directory containing whole-slide images.
        file_type: File extension to glob (e.g. '.tif', '.png').
        out_nm: Name for the output subdirectory created inside WSI_path.
        model: Loaded StarDist model.
        save_tif: Whether to also save the label image as a TIF (~3 GB per WSI).
    """
    WSIs = [os.path.join(WSI_path, f) for f in os.listdir(WSI_path) if f.endswith(file_type)]

    out_pth = os.path.join(WSI_path, out_nm)
    out_pth_json = os.path.join(out_pth, 'json')
    out_pth_tif = os.path.join(out_pth, 'tif')

    os.makedirs(out_pth_json, exist_ok=True)
    if save_tif:
        os.makedirs(out_pth_tif, exist_ok=True)

    physical_devices = tf.config.list_physical_devices('GPU')
    print("Num GPUs Available: ", len(physical_devices))

    for img_pth in WSIs:
        try:
            name = os.path.basename(img_pth)
            json_out = os.path.join(out_pth_json, name[:-len(file_type)] + '.json')

            if os.path.exists(json_out):
                print(f'Skipping {name}')
                continue

            print(f'Starting {name}')
            if 'tif' in file_type:
                img = imread(img_pth)
            else:
                Image.MAX_IMAGE_PIXELS = None
                img = np.array(Image.open(img_pth))
            img = img / 255

            if save_tif:
                labels, polys = model.predict_instances_big(
                    img, axes='YXC', block_size=4096, min_overlap=128, context=128, n_tiles=(4, 4, 1)
                )
                print('Saving json...')
                save_json_from_WSI_pred(polys, out_pth_json, name)
                print('Saving tif...')
                imwrite(os.path.join(out_pth_tif, name[:-5] + '.tif'), labels)
            else:
                _, polys = model.predict_instances_big(
                    img, axes='YXC', block_size=4096, min_overlap=128, context=128, n_tiles=(4, 4, 1)
                )
                print('Saving json...')
                save_json_from_WSI_pred(polys, out_pth_json, name)

        except Exception as e:
            print(f'Error on {img_pth}: {e}')


def augment_tiles(tiles: List[np.ndarray], masks: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Augment H&E tile / mask pairs with rotations and a horizontal flip.

    Produces 8 images per input (original + 3 rotations + flip + 3 rotated flips).
    """
    assert len(tiles) == len(masks)

    HE_aug = [[] for _ in range(len(tiles))]
    mask_aug = [[] for _ in range(len(masks))]

    for i in range(len(tiles)):
        im = Image.fromarray(tiles[i])
        lbl = Image.fromarray(masks[i])

        HE_aug[i].append(im)
        mask_aug[i].append(lbl)

        for _ in range(3):
            im = im.rotate(90)
            HE_aug[i].append(im)
            lbl = lbl.rotate(90)
            mask_aug[i].append(lbl)

        im = Image.fromarray(tiles[i]).transpose(Image.FLIP_LEFT_RIGHT)
        lbl = Image.fromarray(masks[i]).transpose(Image.FLIP_LEFT_RIGHT)
        HE_aug[i].append(im)
        mask_aug[i].append(lbl)

        for _ in range(3):
            im = im.rotate(90)
            HE_aug[i].append(im)
            lbl = lbl.rotate(90)
            mask_aug[i].append(lbl)

    HE_aug = [np.array(i) for x in HE_aug for i in x]
    mask_aug = [np.array(i) for x in mask_aug for i in x]

    return HE_aug, mask_aug


def split_train_val_set(
    tiles: List[np.ndarray], masks: List[np.ndarray], val_ratio: float
) -> Tuple[List, List, List, List]:
    """Randomly split tiles and masks into training and validation sets.

    Args:
        val_ratio: Fraction of tiles reserved for validation (0–1).
    """
    assert len(tiles) == len(masks)

    n_tiles = len(tiles)
    num_val = round(n_tiles * val_ratio)
    val_indices = sorted(random.sample(range(n_tiles), num_val))
    train_indices = sorted(list(set(range(n_tiles)) - set(val_indices)))

    return (
        [tiles[i] for i in train_indices],
        [masks[i] for i in train_indices],
        [tiles[i] for i in val_indices],
        [masks[i] for i in val_indices],
    )


def normalize_images(tiles: List[np.ndarray]) -> List[np.ndarray]:
    """Normalize H&E tiles to [0, 1] by dividing by 255."""
    return [np.divide(tile, 255) for tile in tiles]


def get_loss_data(pth_training_log: str, pth_out: str) -> list:
    """Extract per-epoch loss values from a TensorFlow event log and write to loss.txt."""
    loss_values = []
    for summary in summary_iterator(pth_training_log):
        for value in summary.summary.value:
            if value.tag == 'epoch_loss':
                loss = struct.unpack('f', value.tensor.tensor_content)[0]
                loss_values.append(loss)

    out_txt_name = os.path.join(pth_out, 'loss.txt')
    with open(out_txt_name, 'w') as f:
        f.write('\n'.join(map(str, loss_values)) + '\n')

    return loss_values


def plot_predictions_vs_gt(
    tile: np.ndarray, gt_mask: np.ndarray, pred_mask: np.ndarray, cmap: ListedColormap
) -> None:
    """Show H&E, ground-truth overlay, and predicted overlay in a 1x3 figure."""
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


def save_json_from_WSI_pred(result: dict, out_pth: str, name: str) -> None:
    """Serialize StarDist predict_instances_big output to the custom JSON format.

    Output format per nucleus: {"centroid": [[x, y]], "contour": [[x0,y0], ...]}.
    """
    coords = result['coord']
    points = result['points']
    json_data = []

    for i in range(len(points)):
        centroid = [int(points[i][0]), int(points[i][1])]
        contour = [[float(coord) for coord in xy[::-1]] for xy in coords[i]]
        json_data.append({"centroid": [centroid], "contour": [contour]})

    new_fn = name[:-5] + '.json'
    with open(os.path.join(out_pth, new_fn), 'w') as outfile:
        json.dump(json_data, outfile)
    print('Finished', new_fn)


def format_seg_data(segmentation_data: list, ds: float) -> list:
    """Downsample centroids and contours by factor ds for display or export."""
    data_list = []
    for data in segmentation_data:
        centroid = data['centroid'][0]
        contour = data['contour'][0]
        ds_centroid = [int(c / ds) for c in centroid]
        ds_contour = [[round(x, 2), round(y, 2)] for x, y in zip(
            [v / ds for v in contour[0]], [v / ds for v in contour[1]]
        )]
        data_list.append([ds_centroid, ds_contour])
    return data_list


class TileSetScorer:
    """Score a set of predicted segmentation masks against ground truth.

    Computes per-tile metrics (IoU, TP, FP, FN, Precision, Recall, F1,
    Segmentation Quality, Panoptic Quality) across a range of IoU thresholds (taus).
    Assumes nuclei are mostly convex (centroid lies inside the object).
    """

    def __init__(
        self,
        base_names: List[str],
        gt_set: List[np.ndarray],
        pred_set: List[np.ndarray],
        taus: List[float],
    ):
        self.base_names = base_names
        self.gt_set = gt_set
        self.pred_set = pred_set
        self.taus = taus
        self.df_results_granular = self.score_set()
        self.df_results_summary = self.summarize_scores(self.df_results_granular)

    def score_set(self) -> pd.DataFrame:
        columns = ['Image', 'Tau', 'IoU', 'TP', 'FP', 'FN',
                   'Precision', 'Recall', 'Avg Precision', 'F1 Score', 'Seg Quality', 'Pan Quality']
        df_results = pd.DataFrame(columns=columns)
        for i, base_name in enumerate(self.base_names):
            gt, pred = self.gt_set[i], self.pred_set[i]
            for tau in self.taus:
                results = {'Image': [base_name], 'Tau': [tau]}
                scores = ScoringSubroutine(gt, pred, tau).scores
                for j, score in enumerate(scores):
                    results[columns[j + 2]] = score
                df_results = pd.concat([df_results, pd.DataFrame(results)], axis=0, ignore_index=True)
        return df_results

    @staticmethod
    def summarize_scores(df_granular: pd.DataFrame) -> pd.DataFrame:
        df_summary = df_granular.groupby(['Image']).agg(
            {'IoU': 'median', 'Avg Precision': 'mean'}
        ).reset_index()
        df_summary.columns = ['Image', 'IoU', 'mAP']
        return df_summary


class ScoringSubroutine:
    """Compute detection and segmentation metrics for one tile at one IoU threshold.

    Assumes nuclei are mostly convex so that their centroids fall inside the object mask.
    """

    def __init__(self, gt: np.ndarray, pred: np.ndarray, tau: float):
        gt_centroids = self.find_centroids(gt)
        pred_centroids = self.find_centroids(pred)
        self.scores = self.calculate_scores(gt, pred, tau, gt_centroids, pred_centroids)

    @staticmethod
    def find_centroids(mask: np.ndarray) -> List[List[int]]:
        centroids = []
        for object_id in np.unique(mask)[1:]:
            binary_mask = (mask == object_id)
            x_coords, y_coords = np.where(binary_mask)
            centroids.append([int(np.round(np.mean(x_coords))), int(np.round(np.mean(y_coords)))])
        return centroids

    def calculate_scores(
        self,
        gt: np.ndarray,
        pred: np.ndarray,
        tau: float,
        gt_centroids: List[List[int]],
        pred_centroids: List[List[int]],
    ):
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
    def calc_iou(array1: np.ndarray, array2: np.ndarray) -> float:
        intersection = np.sum(np.logical_and(array1, array2))
        union = np.sum(np.logical_or(array1, array2))
        return intersection / union

    def calc_tp_fp_sg(
        self, gt: np.ndarray, pred: np.ndarray, tau: float, pred_centroids: List[List[int]]
    ) -> Tuple[int, int, float]:
        tp, fp, sum_tp_iou = 0, 0, 0.0
        for x, y in pred_centroids:
            gt_val = gt[x][y]
            pred_val = pred[x][y]
            if gt_val:
                iou = self.calc_iou(gt == gt_val, pred == pred_val)
                if iou >= tau:
                    tp += 1
                    sum_tp_iou += iou
                else:
                    fp += 1
            else:
                fp += 1
        sg = sum_tp_iou / tp if tp > 0 else 0
        return tp, fp, sg

    def calc_fn(
        self, gt: np.ndarray, pred: np.ndarray, tau: float, gt_centroids: List[List[int]]
    ) -> int:
        fn = 0
        for x, y in gt_centroids:
            pred_val = pred[x][y]
            gt_val = gt[x][y]
            if pred_val:
                iou = self.calc_iou(gt == gt_val, pred == pred_val)
                if iou < tau:
                    fn += 1
            else:
                fn += 1
        return fn


def get_stats(
    HE_tiles_pth: str,
    mask_gt_tiles: List[np.ndarray],
    mask_pred_tiles: List[np.ndarray],
    taus: List[float],
) -> pd.DataFrame:
    """Return a DataFrame of per-tile scoring metrics for a set of tiles."""
    nms = [os.path.basename(f) for f in os.listdir(HE_tiles_pth) if f.endswith('.tif')]
    scores = TileSetScorer(nms, mask_gt_tiles, mask_pred_tiles, taus)
    return scores.score_set()


def make_f1_plot(HE_tiles_pth: str, results: pd.DataFrame, taus: List[float]) -> None:
    """Bar chart of F1 scores per tile with a target threshold line."""
    nms = [os.path.basename(f) for f in os.listdir(HE_tiles_pth) if f.endswith('.tif')]
    names = results['Image'].tolist()
    names = [name.split(".")[0][21:] for name in names]
    names = [n[:5] if len(n) > 6 else n for n in names]

    f1_scores = results['F1 Score']
    index = np.arange(len(nms))

    fig = plt.figure(figsize=(25, 10))
    fig.set_facecolor('white')
    plt.bar(index, f1_scores, color='darksalmon')
    plt.xlabel("Tile Name", fontsize=20)
    plt.ylabel("F1 Score", fontsize=20)
    plt.title("F1 Scores in Testing Tiles (tau = 0.7)", fontsize=28)
    plt.axhline(y=0.7, linestyle='--', color='red', label=f'Target F1 = {taus[0]}')
    plt.ylim(0, 1)
    plt.xticks(index, names)
    plt.legend(fontsize=20)
    plt.show()
