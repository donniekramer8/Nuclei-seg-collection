import json
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix, csgraph
import os
from tifffile import imread
import matplotlib.pyplot as plt


def get_geojson_centroids(pth):
    """Returns a list of xy coordinates of centroids from contours in a geojson file"""
    geo_data = json.load(open(pth))
    centroids_ann = []

    for i in range(len(geo_data['features'])):
        data = geo_data['features'][i]
        coords = data['geometry']['coordinates'][0]
        x_cent = 0
        y_cent = 0
        for pair in coords:
            x_cent += pair[1]
            y_cent += pair[0]
        x_cent /= len(coords)
        y_cent /= len(coords)
        centroids_ann.append([x_cent, y_cent])
    
    centroids_ann = np.array(centroids_ann)
    
    return centroids_ann

def get_json_centroids(pth):
    """Returns a list of centroids and contours from a stardist custom output json file"""
    segmentation_data = json.load(open(pth))

    centroids = np.array([nuc['centroid'][0] for nuc in segmentation_data])
    contours = np.array([nuc['contour'] for nuc in segmentation_data])

    return centroids, contours

def colocalize_points(points_a: np.ndarray, points_b: np.ndarray, r: int):
    """ Find pairs that minimize global distance. Filters out anything outside radius `r` """

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(points_b)
    distances, b_indices = neigh.radius_neighbors(points_a, radius=r)

    # flatten and get indices for A. This will also drop points in A with no matches in range
    d_flat = np.hstack(distances) + 1
    b_flat = np.hstack(b_indices)
    a_flat = np.array([i for i, neighbors in enumerate(distances) for n in neighbors])

    # filter out A points that cannot be matched
    sm = csr_matrix((d_flat, (a_flat, b_flat)))
    a_matchable = csgraph.maximum_bipartite_matching(sm, perm_type='column')
    sm_filtered = sm[a_matchable != -1]

    # now run the distance minimizing matching
    row_match, col_match = csgraph.min_weight_full_bipartite_matching(sm_filtered)
    return row_match, col_match

def adjust_contours_match(contours_matched, x, y):

    contours_matched_adjusted = []
    for i in range(len(contours_matched)):
        contour = contours_matched[i][0]
        x_coords = contour[0]
        y_coords = contour[1]
        x_coords = [point-x for point in x_coords]
        y_coords = [point-y for point in y_coords]
        
        shape = list(zip(x_coords, y_coords))
        contours_matched_adjusted.append(shape)
    
    return contours_matched_adjusted

def plot_results(ndpi_pth, cropping, centroids, contours, matching):
    crop_x, crop_y, tile_size = cropping

    indices_not_matched = np.setdiff1d(range(len(centroids)), matching[1])
    indices_matched = matching[1]

    centroids_matched = centroids[indices_matched]
    adj_centroids_matched = [[pair[0] - crop_y, pair[1] - crop_x] for pair in centroids_matched]

    contours_matched = contours[indices_matched]
    contours_not_matched = contours[indices_not_matched]

    contours_matched_adjusted = adjust_contours_match(contours_matched, crop_x, crop_y)
    contours_not_matched_adjusted = adjust_contours_match(contours_not_matched, crop_x, crop_y)

    # flip x and y   >:(   # <- face
    reversed_contours = [[(y, x) for x, y in polygon] for polygon in contours_matched_adjusted]
    reversed_contours_negative = [[(y, x) for x, y in polygon] for polygon in contours_not_matched_adjusted]

    fig, ax = plt.subplots(figsize=(16, 8))

    img = imread(os.path.join(ndpi_pth))

    #print([crop_x, crop_x+tile_size, crop_y, crop_y+tile_size])
    ax.imshow(img[crop_x:crop_x+tile_size,crop_y:crop_y+tile_size])
    ax.set_axis_off()

    # Plot each reversed polygon on the same image
    for polygon in reversed_contours:
        x_coords, y_coords = zip(*polygon)
        x_coords = list(x_coords) + [x_coords[0]]  # Close the polygon
        y_coords = list(y_coords) + [y_coords[0]]  # Close the polygon

        color = 'yellow'

        skip = False
        for x in x_coords:
            if x < 0 or x > (tile_size - 1):
                skip = True
                break
        for y in y_coords:
            if y < 0 or y > (tile_size - 1):
                skip = True
                break
        
        if not skip:

            ax.plot(x_coords, y_coords, alpha=0.3, color=color)
            ax.fill(x_coords, y_coords, alpha=0.3, color=color)  # Fill the polygon
        
    # Plot each reversed polygon on the same image
    for polygon in reversed_contours_negative:
        x_coords, y_coords = zip(*polygon)
        x_coords = list(x_coords) + [x_coords[0]]  # Close the polygon
        y_coords = list(y_coords) + [y_coords[0]]  # Close the polygon

        color = 'red'
        
        skip = False
        for x in x_coords:
            if x < 0 or x > (tile_size - 1):
                skip = True
                break
        for y in y_coords:
            if y < 0 or y > (tile_size - 1):
                skip = True
                break
        
        if not skip:

            ax.plot(x_coords, y_coords, alpha=0.4, color=color)
            ax.fill(x_coords, y_coords, alpha=0.4, color=color)  # Fill the polygon

    # Set labels and title for the plot
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_title('Qupath selected nuclei vs unselected')

    plt.show()

def get_matched_inds(ndpi_pth, centroids, contours, matching):
    centroids_all = [centroids[i] for i in centroids]
    centroids_matched = [pair.tolist() for pair in matching[1]]

    indices_not_matched = np.setdiff1d(matching[1], range(len(centroids)))
    indices_matched = matching[1]
    
    return indices_matched

def save_json_data_from_selected(coords, points, out_pth, name):
    """Saves a json file with centroids and contours for StarDist output."""

    new_fn = name[:-5] + '.json'
    out_nm = os.path.join(out_pth, new_fn)
    if not os.path.exists(out_nm):

        json_data = []

        for i in range(len(points)):
            point = points[i]
            contour = coords[i]
            #print(contour)
            centroid = [int(point[0]), int(point[1])]  # TODO: FIX
            contour = [[coord for coord in xy] for xy in contour][0]

            # Create a new dictionary for each contour
            dict_data = {
                "centroid": [centroid],
                "contour": [contour]
            }

            json_data.append(dict_data)

        with open(out_nm,'w') as outfile:
            json.dump(json_data, outfile)
        print('Finished',new_fn)
    else:
        print(f'{os.path.basename(out_nm)} already exists, skipping...')

    