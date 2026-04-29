import copy
import json
import os
import pickle
from typing import Optional

import cv2
import numpy as np
import pandas as pd
from scipy.io import savemat
from tifffile import imread


def _cntarea(cnt: np.ndarray) -> float:
    return cv2.contourArea(np.array(cnt).astype(np.float32))


def _cntperi(cnt: np.ndarray) -> float:
    return cv2.arcLength(np.array(cnt).astype(np.float32), True)


def _cntMA(cnt: np.ndarray):
    """Fit an ellipse and return (major_axis, minor_axis, orientation)."""
    [(x, y), (MA, ma), orientation] = cv2.fitEllipse(np.array(cnt).astype(np.float32))
    return np.max((MA, ma)), np.min((MA, ma)), orientation


def _fix_contours(contours: list) -> np.ndarray:
    """Reformat raw JSON contour lists into OpenCV-compatible int32 arrays."""
    contours_fixed = []
    for polygon in contours:
        coords = np.array([list(zip(x, y)) for x, y in [polygon[0]]][0], dtype=np.int32)
        contours_fixed.append(coords)
    return np.array(contours_fixed)


def _adjust_contours(contour: np.ndarray, crop_x: int, crop_y: int) -> np.ndarray:
    """Translate contour coordinates relative to a crop origin."""
    contour = contour.copy()
    for i, xy in enumerate(contour):
        contour[i] = [xy[0] - crop_x, xy[1] - crop_y]
    return contour


def _get_rgb_avg(centroid, contour_raw, offset: int, image: np.ndarray):
    """Extract mean and std R/G/B intensities within a nucleus contour.

    Crops a (2*offset) x (2*offset) patch around the centroid, draws the
    contour as a filled mask, and computes per-channel statistics.
    Returns -1 for all values if the centroid is too close to the image edge.
    """
    x_low = centroid[0] - offset
    x_high = centroid[0] + offset
    y_low = centroid[1] - offset
    y_high = centroid[1] + offset

    h, w = image.shape[:2]
    if offset > centroid[0] or offset > centroid[1] or centroid[0] > (h - offset) or centroid[1] > (w - offset):
        return -1, -1, -1, -1, -1, -1

    im_crop = np.array(image[x_low:x_high, y_low:y_high], dtype=np.uint16)
    crop_x = centroid[0] - offset - 1
    crop_y = centroid[1] - offset - 1

    contour_adj = _adjust_contours(contour_raw, crop_x, crop_y)
    # StarDist stores contours as (row, col) — flip to (x, y) for OpenCV
    rev_contour = contour_adj[:, [1, 0]]

    mask = np.zeros_like(im_crop[:, :, 0], dtype=np.uint16)
    cv2.drawContours(mask, [rev_contour], 0, 1, thickness=cv2.FILLED)

    num_pixels = np.count_nonzero(mask)
    if num_pixels == 0:
        return -1, -1, -1, -1, -1, -1

    r = im_crop[:, :, 0] * mask
    g = im_crop[:, :, 1] * mask
    b = im_crop[:, :, 2] * mask

    return (
        round(np.sum(r) / num_pixels, 2),
        round(np.sum(g) / num_pixels, 2),
        round(np.sum(b) / num_pixels, 2),
        float(np.std(r)),
        float(np.std(g)),
        float(np.std(b)),
    )


def _extract_features_for_slide(segmentation_data: list, image: np.ndarray, slide_id: str) -> pd.DataFrame:
    """Compute all morphology and color features for every nucleus in one slide."""
    centroids = [nuc['centroid'][0] for nuc in segmentation_data]
    contours = [nuc['contour'] for nuc in segmentation_data]
    contours_fixed = _fix_contours(contours)
    contours_np = np.array(contours)
    np_centroids = np.array(centroids)

    offset = 30

    rows = []
    for j in range(len(contours_fixed)):
        centroid = centroids[j]
        contour_raw = copy.copy(contours_fixed[j])
        r_avg, g_avg, b_avg, r_std, g_std, b_std = _get_rgb_avg(centroid, contour_raw, offset, image)

        contour = contours_np[j][0].transpose()
        area = _cntarea(contour)
        perimeter = _cntperi(contour)
        circularity = 4 * np.pi * area / perimeter ** 2
        MA, ma, orientation = _cntMA(contour)
        aspect_ratio = MA / ma

        dists = np.linalg.norm(contour - centroid, axis=1)

        rows.append({
            'Centroid_x': np_centroids[j, 1],
            'Centroid_y': np_centroids[j, 0],
            'Area': area,
            'Perimeter': perimeter,
            'Circularity': circularity,
            'Aspect Ratio': aspect_ratio,
            'compactness': perimeter ** 2 / area,
            'eccentricity': float(np.sqrt(1 - (ma / MA) ** 2)),
            'extent': area / (MA * ma),
            'form_factor': (perimeter ** 2) / (4 * np.pi * area),
            'maximum_radius': float(np.max(dists)),
            'mean_radius': float(np.mean(dists)),
            'median_radius': float(np.median(dists)),
            'minor_axis_length': ma,
            'major_axis_length': MA,
            'orientation_degrees': float(np.degrees(orientation)),
            'r_mean_intensity': r_avg,
            'g_mean_intensity': g_avg,
            'b_mean_intensity': b_avg,
            'r_std': r_std,
            'g_std': g_std,
            'b_std': b_std,
            'slide_num': slide_id,
        })

    return pd.DataFrame(rows).astype(np.float32)


def get_json_file_list(WSI_path: str, json_folder_name: str) -> list:
    """Return a sorted list of JSON segmentation file paths."""
    out_pth_json = os.path.join(WSI_path, json_folder_name, 'json')
    return sorted([
        os.path.join(out_pth_json, f)
        for f in os.listdir(out_pth_json)
        if f.endswith(".json")
    ])


def write_df_features_pkl(WSI_path: str, out_name: str, WSI_file_type: str) -> None:
    """Extract nuclear morphology features from all slides and save as .pkl files.

    Expects paired WSI images and JSON segmentation files. Output .pkl files
    (one per slide) are written to <out_name>/json/nuclear_morph_features_pkl/.

    Args:
        WSI_path: Directory containing the whole-slide images.
        out_name: Subdirectory name used during segmentation (contains the json/ folder).
        WSI_file_type: File extension of the WSIs (e.g. '.tif').
    """
    WSI_full_pth_list = sorted([
        os.path.join(WSI_path, f)
        for f in os.listdir(WSI_path)
        if f.endswith(WSI_file_type)
    ])
    json_full_pth_list = get_json_file_list(WSI_path, out_name)

    outpth = os.path.join(os.path.dirname(json_full_pth_list[0]), 'nuclear_morph_features_pkl')
    os.makedirs(outpth, exist_ok=True)

    for i, json_f_name in enumerate(json_full_pth_list):
        nm = os.path.basename(json_f_name).split('.')[0]
        outnm = os.path.join(outpth, f'{nm}.pkl')
        print(nm)

        if os.path.exists(outnm):
            print('skipping')
            continue

        try:
            with open(json_f_name) as f:
                segmentation_data = json.load(f)
        except Exception:
            print(f'error reading json... Skipping {json_f_name}')
            continue

        image = imread(WSI_full_pth_list[i])
        df = _extract_features_for_slide(segmentation_data, image, slide_id=nm[-4:])
        df.to_pickle(outnm)


def write_df_features_pkl_single(json_file_pth: str, WSI_file_pth: str, outpth: str) -> None:
    """Extract features for a single slide and save as a .pkl file.

    Args:
        json_file_pth: Path to the JSON segmentation file for this slide.
        WSI_file_pth: Path to the corresponding whole-slide image.
        outpth: Directory where the .pkl will be written.
    """
    nm = os.path.basename(json_file_pth)[:-5]
    outnm = os.path.join(outpth, f'{nm}.pkl')
    print(outnm)

    if os.path.exists(outnm):
        print('skipping')
        return

    with open(json_file_pth) as f:
        segmentation_data = json.load(f)

    image = imread(WSI_file_pth)
    df = _extract_features_for_slide(segmentation_data, image, slide_id=nm[-4:])
    df.to_pickle(outnm)


def write_mat_features_from_pkl(WSI_path: str, out_name: str) -> None:
    """Convert .pkl feature DataFrames to MATLAB .mat files.

    Reads from <out_name>/json/nuclear_morph_features_pkl/ and writes
    to a nuclear_morph_features_mat/ subdirectory alongside it.
    """
    pkl_pth = os.path.join(WSI_path, out_name, 'json', 'nuclear_morph_features_pkl')
    mat_pth = os.path.join(pkl_pth, 'nuclear_morph_features_mat')
    os.makedirs(mat_pth, exist_ok=True)

    dfs = [os.path.join(pkl_pth, f) for f in os.listdir(pkl_pth) if f.endswith('.pkl')]

    for dfnm in dfs:
        outnm = os.path.join(mat_pth, os.path.basename(dfnm)[:-4] + '.mat')
        print("Saving:", dfnm)

        with open(dfnm, 'rb') as f:
            df = pickle.load(f)

        col_names = df.columns.tolist()
        mat_array = df.to_numpy()
        savemat(outnm, {'features': mat_array, 'feature_names': col_names})
