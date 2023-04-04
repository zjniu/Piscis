import numpy as np

from skimage import feature, filters, measure


def log_filter(image, sigma):

    image = filters.gaussian(image, sigma)
    image = filters.laplace(image)

    return image


def compute_spot_coordinates(image, threshold, min_distance):

    stack = image.ndim == 3

    if stack:
        labels = measure.label(image > threshold)
        coords = np.array([region['centroid'] for region in measure.regionprops(labels)], dtype=int)
    else:
        coords = feature.peak_local_max(image, threshold_abs=threshold, min_distance=min_distance, exclude_border=False)

    if len(coords) == 0:
        coords = np.empty((0, image.ndim), dtype=np.float32)

    return coords
