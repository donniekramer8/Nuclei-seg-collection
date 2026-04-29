import json
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix, csgraph
from sklearn.neighbors import NearestNeighbors
from tifffile import imread


def get_geojson_centroids(pth: str) -> np.ndarray:
    """Return (N, 2) array of [x, y] centroids computed from GeoJSON polygon coordinates."""
    geo_data = json.load(open(pth))
    centroids = []
    for feature in geo_data['features']:
        coords = feature['geometry']['coordinates'][0]
        x_cent = sum(p[1] for p in coords) / len(coords)
        y_cent = sum(p[0] for p in coords) / len(coords)
        centroids.append([x_cent, y_cent])
    return np.array(centroids)


def get_json_centroids(pth: str):
    """Return centroids and contours from a custom StarDist JSON output file.

    Returns:
        centroids: (N, 2) array of [row, col] centroid coordinates.
        contours: array of raw contour data.
    """
    segmentation_data = json.load(open(pth))
    centroids = np.array([nuc['centroid'][0] for nuc in segmentation_data])
    contours = np.array([nuc['contour'] for nuc in segmentation_data])
    return centroids, contours


def colocalize_points(points_a: np.ndarray, points_b: np.ndarray, r: int):
    """Match two point sets by minimizing global pairwise distance within radius r.

    Uses bipartite matching so each point is matched at most once.

    Args:
        points_a: (N, 2) query points.
        points_b: (M, 2) reference points.
        r: Maximum allowable distance for a match.

    Returns:
        row_match, col_match: Indices into points_a and points_b of matched pairs.
    """
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(points_b)
    distances, b_indices = neigh.radius_neighbors(points_a, radius=r)

    d_flat = np.hstack(distances) + 1
    b_flat = np.hstack(b_indices)
    a_flat = np.array([i for i, neighbors in enumerate(distances) for _ in neighbors])

    sm = csr_matrix((d_flat, (a_flat, b_flat)))
    a_matchable = csgraph.maximum_bipartite_matching(sm, perm_type='column')
    sm_filtered = sm[a_matchable != -1]

    row_match, col_match = csgraph.min_weight_full_bipartite_matching(sm_filtered)
    return row_match, col_match


def adjust_contours_match(contours_matched: np.ndarray, x: int, y: int) -> list:
    """Translate contours by (x, y) to convert from WSI coordinates to crop coordinates."""
    result = []
    for contour in contours_matched:
        pts = contour[0]
        shape = list(zip([p - x for p in pts[0]], [p - y for p in pts[1]]))
        result.append(shape)
    return result


def plot_results(ndpi_pth: str, cropping: tuple, centroids: np.ndarray, contours: np.ndarray, matching: tuple) -> None:
    """Visualize matched (yellow) and unmatched (red) nuclei contours over a WSI crop.

    Args:
        ndpi_pth: Path to the whole-slide image.
        cropping: (crop_x, crop_y, tile_size) defining the region to display.
        centroids: All nucleus centroids in WSI coordinates.
        contours: All nucleus contours in WSI coordinates.
        matching: (row_match, col_match) from colocalize_points.
    """
    crop_x, crop_y, tile_size = cropping
    indices_matched = matching[1]
    indices_not_matched = np.setdiff1d(range(len(centroids)), indices_matched)

    contours_matched_adj = adjust_contours_match(contours[indices_matched], crop_x, crop_y)
    contours_not_matched_adj = adjust_contours_match(contours[indices_not_matched], crop_x, crop_y)

    reversed_matched = [[(y, x) for x, y in poly] for poly in contours_matched_adj]
    reversed_not_matched = [[(y, x) for x, y in poly] for poly in contours_not_matched_adj]

    fig, ax = plt.subplots(figsize=(16, 8))
    img = imread(ndpi_pth)
    ax.imshow(img[crop_x:crop_x + tile_size, crop_y:crop_y + tile_size])
    ax.set_axis_off()

    for color, polygons in [('yellow', reversed_matched), ('red', reversed_not_matched)]:
        for polygon in polygons:
            xs, ys = zip(*polygon)
            xs = list(xs) + [xs[0]]
            ys = list(ys) + [ys[0]]
            if any(v < 0 or v > tile_size - 1 for v in xs + ys):
                continue
            ax.plot(xs, ys, alpha=0.4, color=color)
            ax.fill(xs, ys, alpha=0.4, color=color)

    ax.set_title('QuPath selected nuclei (yellow) vs unselected (red)')
    plt.show()


def get_matched_inds(matching: tuple) -> np.ndarray:
    """Return the matched indices from a colocalize_points result."""
    return matching[1]


def save_json_data_from_selected(coords, points, out_pth: str, name: str) -> None:
    """Save a subset of nuclei (selected centroids and contours) to a JSON file.

    Skips writing if the output file already exists.
    """
    new_fn = name[:-5] + '.json'
    out_nm = os.path.join(out_pth, new_fn)

    if os.path.exists(out_nm):
        print(f'{os.path.basename(out_nm)} already exists, skipping...')
        return

    json_data = []
    for i in range(len(points)):
        centroid = [int(points[i][0]), int(points[i][1])]
        contour = [[coord for coord in xy] for xy in coords[i]][0]
        json_data.append({"centroid": [centroid], "contour": [contour]})

    with open(out_nm, 'w') as outfile:
        json.dump(json_data, outfile)
    print('Finished', new_fn)
