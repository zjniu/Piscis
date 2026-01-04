import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from numba import njit
from scipy import optimize
from skimage import feature, measure
from typing import Dict, List, Sequence, Tuple


def compute_spot_coordinates(
        labels: np.ndarray,
        deltas: np.ndarray,
        threshold: float,
        min_distance: int,
) -> np.ndarray:

    """Compute spot coordinates from labels and deltas.

    Parameters
    ----------
    labels : np.ndarray
        Labels.
    deltas : np.ndarray
        Displacement vectors.
    threshold : float
        Detection threshold between 0 and 1.
    min_distance : int
        Minimum distance between spots.

    Returns
    -------
    coords : np.ndarray
        Coordinates of detected spots.
    """

    # Check if the labels are in a stack.
    stack = labels.ndim == 3
    if stack:

        # Use connected components to detect spots if labels are in a stack.
        labels = measure.label(labels > threshold)
        peaks = np.array([region['centroid'] for region in measure.regionprops(labels)], dtype=int)

    else:

        # Use peak local maxima to detect spots if labels are not in a stack.
        peaks = feature.peak_local_max(labels,
                                       min_distance=min_distance, threshold_abs=threshold, exclude_border=False)

    # Apply deltas to detected spots.
    if len(peaks) > 0:
        if stack:
            coords = peaks + np.pad(deltas[peaks[:, 0], :, peaks[:, 1], peaks[:, 2]], ((0, 0), (1, 0)))
        else:
            coords = peaks + deltas[:, peaks[:, 0], peaks[:, 1]].T
    else:
        coords = np.empty((0, labels.ndim), dtype=np.float32)

    return coords


def deformable_max_pool(
        labels: torch.Tensor,
        deltas: torch.Tensor,
        kernel_size: Sequence[int] = (3, 3)
) -> torch.Tensor:

    """Max pool labels using deltas.

    Parameters
    ----------
    labels : torch.Tensor
        Labels.
    deltas : torch.Tensor
        Displacement vectors.
    kernel_size : Sequence[int], optional
        Kernel size or window size of the max pooling operation. Default is (3, 3).

    Returns
    -------
    pooled_labels : torch.Tensor
        Pooled labels.
    """

    h, w = labels.shape
    kh, kw = kernel_size
    device = labels.device
    padding = (kh // 2, kw // 2)
    patch_size = kh * kw

    # Generate an index map.
    i = torch.arange(h, device=device)
    j = torch.arange(w, device=device)
    ii, jj = torch.meshgrid(i, j, indexing='ij')
    index_map = torch.stack((ii, jj))

    # Compute the pixel convergence array after applying deltas.
    convergence = torch.round(deltas + index_map)

    # Max pool the label values of pixels that converge at each pixel location.
    unfold = nn.Unfold(kernel_size=kernel_size, padding=padding)
    convergence = unfold(convergence[None]).view(2, patch_size, h, w).int()
    labels = unfold(labels[None, None]).view(patch_size, h, w)
    matches = (convergence == index_map[:, None]).all(dim=0)
    pooled_labels = (labels * matches).max(dim=0).values

    return pooled_labels


vmap_deformable_max_pool = torch.vmap(deformable_max_pool, in_dims=(0, 0, None))


def deformable_softmax_pool(
        labels: torch.Tensor,
        deltas: torch.Tensor,
        x: torch.Tensor = None,
        kernel_size: Sequence[int] = (3, 3),
        temperature: float = 0.05
) -> torch.Tensor:

    """Softmax pool labels or another tensor using deltas.

    Parameters
    ----------
    labels : torch.Tensor
        Labels.
    deltas : torch.Tensor
        Displacement vectors.
    x : torch.Tensor, optional
        Tensor to be pooled. Default is None.
    kernel_size : Sequence[int], optional
        Kernel size or window size of the softmax pooling operation. Default is (3, 3).
    temperature : float
        Temperature parameter for softmax. Default is 0.05.

    Returns
    -------
    pooled_labels : torch.Tensor
        Pooled labels.
    """

    h, w = labels.shape
    kh, kw = kernel_size
    device = labels.device
    padding = (kh // 2, kw // 2)
    patch_size = kh * kw

    # Generate an index map.
    i = torch.arange(h, device=device)
    j = torch.arange(w, device=device)
    ii, jj = torch.meshgrid(i, j, indexing='ij')
    index_map = torch.stack((ii, jj))

    # Compute the pixel convergence array after applying deltas.
    convergence = torch.round(deltas + index_map)

    # Softmax pool the label values of pixels that converge at each pixel location.
    unfold = nn.Unfold(kernel_size=kernel_size, padding=padding)
    convergence = unfold(convergence[None]).view(2, patch_size, h, w).int()
    labels = unfold(labels[None, None]).view(patch_size, h, w)
    if x is None:
        x = labels
    else:
        x = unfold(x[None, None]).view(patch_size, h, w)
    matches = (convergence == index_map[:, None]).all(dim=0)
    labels = labels * matches
    weights = F.softmax(labels / temperature, dim=0)
    pooled_x = (x * weights).sum(dim=0)

    return pooled_x


vmap_deformable_softmax_pool = torch.vmap(deformable_softmax_pool, in_dims=(0, 0, 0, None, None))


def deformable_sum_pool(
        labels: torch.Tensor,
        deltas: torch.Tensor,
        kernel_size: Sequence[int] = (3, 3)
) -> torch.Tensor:

    """Sum pool labels using deltas.

    Parameters
    ----------
    labels : torch.Tensor
        Labels.
    deltas : torch.Tensor
        Displacement vectors.
    kernel_size : Sequence[int], optional
        Kernel size or window size of the sum pooling operation. Default is (3, 3).

    Returns
    -------
    pooled_labels : torch.Tensor
        Pooled labels.
    """

    h, w = labels.shape
    kh, kw = kernel_size
    device = labels.device
    padding = (kh // 2, kw // 2)
    patch_size = kh * kw

    # Generate an index map.
    i = torch.arange(h, device=device)
    j = torch.arange(w, device=device)
    ii, jj = torch.meshgrid(i, j, indexing='ij')
    index_map = torch.stack((ii, jj))

    # Compute the pixel convergence array after applying deltas.
    convergence = torch.round(deltas + index_map)

    # Sum pool the label values of pixels that converge at each pixel location.
    unfold = nn.Unfold(kernel_size=kernel_size, padding=padding)
    convergence = unfold(convergence[None]).view(2, patch_size, h, w).int()
    labels = unfold(labels[None, None]).view(patch_size, h, w)
    matches = (convergence == index_map[:, None]).all(dim=0)
    pooled_labels = (labels * matches).sum(dim=0)

    return pooled_labels


vmap_deformable_sum_pool = torch.vmap(deformable_sum_pool, in_dims=(0, 0, None))


def peak_local_softmax(
        labels: torch.Tensor,
        kernel_size: Sequence[int] = (3, 3),
        temperature: float = 0.05,
) -> torch.Tensor:
    
    """Smooth variant of `peak_local_max` with softmax pooling to find peaks in labels.

    Parameters
    ----------
    labels : torch.Tensor
        Labels.
    kernel_size : Sequence[int], optional
        Kernel size or window size of the softmax pooling operation. Default is (3, 3).
    temperature : float
        Temperature parameter for softmax. Default is 0.05.

    Returns
    -------
    peaked_labels : torch.Tensor
        Peaked labels.
    """

    h, w = labels.shape
    kh, kw = kernel_size
    padding = (kh // 2, kw // 2)

    # Find peaks in labels via softmax-weighted pooling.
    unfold = nn.Unfold(kernel_size=kernel_size, padding=padding)
    labels = unfold(labels[None, None]).view(kh * kw, h, w)
    weights = F.softmax(labels / temperature, dim=0)
    center_index = kw * padding[0] + padding[1]
    peaked_labels = labels[center_index] * weights[center_index]

    return peaked_labels


vmap_peak_local_softmax = torch.vmap(peak_local_softmax, in_dims=(0, None, None))


def pad_and_stack(images: Sequence[np.ndarray]) -> np.ndarray:

    """Pad and stack images.

    Parameters
    ----------
    images : Sequence[np.ndarray]
        List of images to pad and stack.

    Returns
    -------
    stacked_images : np.ndarray
        Stacked images.
    """

    padded_images = pad(images)
    stacked_images = np.stack(padded_images)

    return stacked_images


def pad(images: Sequence[np.ndarray]) -> Sequence[np.ndarray]:

    """Pad images to the same size.

    Parameters
    ----------
    images : Sequence[np.ndarray]
        List of images to pad.

    Returns
    -------
    padded_images : Sequence[np.ndarray]
        List of padded images.
    """

    # Compute the padded image size.
    padded_size = [max([image.shape[i] for image in images]) for i in range(images[0].ndim)]

    # Pad images.
    padded_images = []
    for image in images:
        if image.shape == padded_size:
            padded_image = image
        else:
            pad_width = [(0, m - n) for m, n in zip(padded_size, image.shape)]
            padded_image = np.pad(image, pad_width)
        padded_images.append(padded_image)

    return padded_images


def remove_duplicate_coords(
        coords: np.ndarray,
        threshold: int = 1
) -> np.ndarray:

    """Remove duplicate coordinates within a distance threshold.

    Parameters
    ----------
    coords : np.ndarray
        Coordinates.
    threshold : int, optional
        Distance threshold. Default is 1.

    Returns
    -------
    new_coords : np.ndarray
        Coordinates without duplicates.
    """

    # Create a dictionary to keep track of sets of duplicate coordinates.
    sets = {}

    # Create a dictionary to keep track of the set id of each coordinate.
    coord_set_ids = {i: None for i in range(len(coords))}

    # Create a list to keep track of coordinates that have been checked.
    checked = []

    # Loop through each coordinate and check if it has already been assigned to a set.
    for i, set_id in coord_set_ids.items():

        # If the coordinate has not been assigned to a set yet.
        if set_id is None:

            # Create a new set and add the coordinate to it.
            set_id = len(sets)
            coord_set_ids[i] = set_id
            new_set = [i]
            sets[set_id] = new_set

            # Find other coordinates within the distance threshold and add them to the same set.
            matches = _match_coords(coords, i, set_id, coord_set_ids, checked, threshold)
            new_set.extend(matches)
            to_check = matches.copy()

            # Construct cluster of neighboring coordinates.
            while len(to_check) > 0:

                # Loop through new coordinates and find other coordinates within the distance threshold.
                for j in to_check:
                    matches = _match_coords(coords, j, set_id, coord_set_ids, checked, threshold)
                    to_check.remove(j)
                    new_matches = set(matches) - set(new_set)
                    new_set.extend(new_matches)
                    to_check.extend(new_matches)

                to_check = list(set(to_check) - set(checked))

    # Compute the mean of coordinates in each set.
    new_coords = []
    for s in sets.values():
        old_coords = coords[list(s)]
        new_coords.append(np.mean(old_coords, axis=0))
    new_coords = np.array(new_coords)

    return new_coords


def _match_coords(
        coords: np.ndarray,
        i: int,
        set_id: int,
        coord_set_ids: Dict[int, int],
        checked: List[int],
        threshold: int
) -> List[int]:

    """Match coordinates within a distance threshold and assign them to the same set.

    Parameters
    ----------
    coords : np.ndarray
        Coordinates.
    i : int
        Coordinate index.
    set_id : int
        Set id.
    coord_set_ids : Dict[int, int]
        Dictionary to keep track of the set id of each coordinate.
    checked : List[int]
        List to keep track of coordinates that have been checked.
    threshold : int
        Distance threshold.

    Returns
    -------
    matches : List[int]
        List of matched coordinates.
    """

    # Mean the distances between the ith coordinate and all other coordinates.
    distances = np.sqrt(np.sum((coords[i] - coords) ** 2, axis=1))

    # Find coordinates within the distance threshold.
    matches = list(np.where(distances < threshold)[0])

    # Remove the ith coordinate itself from the matches.
    matches.remove(i)

    # Assign the same set id to all matched coordinates.
    for match in matches:
        coord_set_ids[match] = set_id

    # Add the ith coordinate to the list of checked coordinates.
    checked.append(i)

    return matches


def snap_coords(
        coords: np.ndarray,
        image: np.ndarray,
        window_size: int = 3
) -> np.ndarray:

    """Snap each coordinate to the local maxima of an image in a window around the coordinate.

    Parameters
    ----------
    coords : np.ndarray
        Coordinates.
    image : np.ndarray
        Image.
    window_size : int, optional
        Window size. Must be an odd integer. Default is 3.

    Returns
    -------
    snapped_coords : np.ndarray
        Snapped coordinates.

    Raises
    ------
    ValueError
        If `window_size` is not an odd integer.
    """

    # Check the window size.
    if window_size % 2 == 0:
        raise ValueError("window_size must be an odd integer.")

    # Remove coordinates outside the image.
    coords = coords[(coords[:, 0] > -0.5) & (coords[:, 1] > -0.5) &
                    (coords[:, 0] < image.shape[0] - 0.5) & (coords[:, 1] < image.shape[1] - 0.5)]

    # Create a list to store the snapped coordinates.
    snapped_coords = []

    # Pad the image.
    delta = round((window_size - 1) / 2)
    image = np.pad(image, ((delta, delta), (delta, delta)))

    for coord in coords:

        # Crop the image around the coordinate.
        index = np.rint(coord).astype(int)
        cropped_image = image[index[0]: index[0] + window_size, index[1]: index[1] + window_size]

        # Find the local maximum.
        maximum = np.where(cropped_image == np.max(cropped_image))
        maximum = np.array([maximum[0][0], maximum[1][0]])

        # Snap the coordinate to the local maximum.
        offset = maximum - delta
        snapped_coords.append(index + offset)

    snapped_coords = np.array(snapped_coords)

    return snapped_coords


def fit_coords(
        coords: np.ndarray,
        image: np.ndarray,
        window_size: int = 3,
        max_gaussian_amplitude: float = 2.0,
        keep_failed_fits: bool = False
) -> np.ndarray:

    """Fit a Gaussian to the image in a window around each coordinate and return the center of the Gaussian.

    Parameters
    ----------
    coords : np.ndarray
        Coordinates.
    image : np.ndarray
        Image.
    window_size : int, optional
        Window size. Must be an odd integer. Default is 3.
    max_gaussian_amplitude : float, optional
        Maximum amplitude of the Gaussian used to fit the normalized image within a window. Default is 2.0.
    keep_failed_fits : bool, optional
        Whether to keep the original coordinates when the Gaussian fit fails. Default is False.

    Returns
    -------
    fitted_coords : np.ndarray
        Fitted coordinates.

    Raises
    ------
    ValueError
        If `window_size` is not an odd integer.
    """

    # Check the window size.
    if window_size % 2 == 0:
        raise ValueError("window_size must be an odd integer.")

    # Remove coordinates outside the image.
    coords = coords[(coords[:, 0] > -0.5) & (coords[:, 1] > -0.5) &
                    (coords[:, 0] < image.shape[0] - 0.5) & (coords[:, 1] < image.shape[1] - 0.5)]

    # Create a list to store the fitted coordinates.
    fitted_coords = []

    # Pad the image.
    delta = round((window_size - 1) / 2)
    image = np.pad(image, ((delta, delta), (delta, delta)), mode='reflect')

    # Create a meshgrid for the Gaussian fit.
    x = np.arange(-delta, delta + 1)
    x, y = np.meshgrid(x, x)

    # Define initial parameters and bounds for the Gaussian fit.
    p0 = (1, 0, 0, delta, delta, 0, 0)
    bounds = ((0, -delta, -delta, 0, 0, -np.pi, 0),
              (max_gaussian_amplitude, delta, delta, np.inf, np.inf, np.pi, max_gaussian_amplitude))

    for coord in coords:

        # Crop the image around the coordinate.
        index = np.rint(coord).astype(int)
        cropped_image = image[index[0]: index[0] + window_size, index[1]: index[1] + window_size].ravel()

        # Normalize the cropped image.
        image_min = np.min(cropped_image)
        image_max = np.max(cropped_image)
        cropped_image = (cropped_image - image_min) / (image_max - image_min + 1e-7)

        # Fit a Gaussian to the cropped image.
        try:
            popt, pcov = optimize.curve_fit(_gaussian, (x, y), cropped_image, p0=p0, bounds=bounds)
            fitted_coords.append(index + np.array([popt[2], popt[1]]))
        except RuntimeError:
            if keep_failed_fits:
                fitted_coords.append(coord)

    fitted_coords = np.array(fitted_coords)

    return fitted_coords


@njit
def _gaussian(
        xy: Tuple[np.ndarray, np.ndarray],
        amplitude: float,
        x0: float,
        y0: float,
        sigma_x: float,
        sigma_y: float,
        theta: float,
        offset: float
) -> np.ndarray:

    """2D Gaussian function.

    Parameters
    ----------
    xy : Tuple[np.ndarray, np.ndarray]
        x and y values at which to evaluate the Gaussian.
    amplitude : float
        Amplitude of the Gaussian.
    x0 : float
        x-coordinate of the center of the Gaussian.
    y0 : float
        y-coordinate of the center of the Gaussian.
    sigma_x : float
        Standard deviation of the Gaussian in the x-direction.
    sigma_y : float
        Standard deviation of the Gaussian in the y-direction.
    theta : float
        Rotation angle of the Gaussian.
    offset : float
        Offset of the Gaussian in the z-direction.

    Returns
    -------
    g : np.ndarray
        Gaussian at the given x and y values.
    """

    x, y = xy
    a = (np.cos(theta) ** 2) / (2 * sigma_x ** 2) + (np.sin(theta) ** 2) / (2 * sigma_y ** 2)
    b = -(np.sin(2 * theta)) / (4 * sigma_x ** 2) + (np.sin(2 * theta))/(4 * sigma_y ** 2)
    c = (np.sin(theta) ** 2)/(2 * sigma_x ** 2) + (np.cos(theta) ** 2)/(2 * sigma_y ** 2)
    g = offset + amplitude * np.exp(-(a * ((x - x0) ** 2) + 2 * b * (x - x0) * (y - y0) + c * ((y - y0) ** 2)))
    g = g.ravel()

    return g
