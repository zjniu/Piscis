import jax
import jax.numpy as jnp
import numpy as np

from jax import vmap
from jax.lax import dynamic_slice, scan
from numba import njit
from scipy import optimize
from skimage import feature, measure
from typing import Dict, List, Sequence, Tuple


def compute_spot_coordinates(
        deltas: np.ndarray,
        pooled_labels: np.ndarray,
        threshold: float,
        min_distance: int,
) -> np.ndarray:

    """Compute spot coordinates from deltas and labels.

    Parameters
    ----------
    deltas : np.ndarray
        Displacement vectors.
    pooled_labels : np.ndarray
        Pooled labels.
    threshold : float
        Detection threshold between 0 and 1.
    min_distance : int
        Minimum distance between spots.

    Returns
    -------
    coords : np.ndarray
        Coordinates of detected spots.
    """

    # Check if the pooled labels are in a stack.
    stack = pooled_labels.ndim == 3

    if stack:

        # Use connected components to detect spots if pooled labels are in a stack.
        labels = measure.label(pooled_labels > threshold)
        peaks = np.array([region['centroid'] for region in measure.regionprops(labels)], dtype=int)

    else:

        # Use peak local maxima to detect spots if pooled labels are not in a stack.
        peaks = feature.peak_local_max(pooled_labels,
                                       min_distance=min_distance, threshold_abs=threshold, exclude_border=False)

    # Apply deltas to detected spots.
    if len(peaks) > 0:
        if stack:
            coords = peaks + np.pad(deltas[peaks[:, 0], peaks[:, 1], peaks[:, 2]], ((0, 0), (1, 0)))
        else:
            coords = peaks + deltas[peaks[:, 0], peaks[:, 1]]
    else:
        coords = np.empty((0, pooled_labels.ndim), dtype=np.float32)

    return coords


def scanned_sum_pool(
        deltas: jax.Array,
        labels: jax.Array,
        n_iter: int,
        kernel_size: Sequence[int] = (3, 3)
) -> jax.Array:

    """Scanned version of `sum_pool`.

    Parameters
    ----------
    deltas : jax.Array
        Displacement vectors.
    labels : jax.Array
        Binary labels.
    n_iter : int
        Number of iterations.
    kernel_size : Sequence[int], optional
        Kernel size or window size of the sum pooling operation. Default is (3, 3).

    Returns
    -------
    pooled_labels : jax.Array
        Pooled labels after sum pooling for `n_iter` iterations.
    """

    pooled_labels = scan(lambda c, x: _scanned_sum_pool(deltas, c, kernel_size), labels, jnp.empty(n_iter))[0]

    return pooled_labels


vmap_scanned_sum_pool = vmap(scanned_sum_pool, in_axes=(0, 0, None, None))


def _scanned_sum_pool(
        deltas: jax.Array,
        pooled_labels: jax.Array,
        kernel_size: Sequence[int]
) -> Tuple[jax.Array, None]:

    """Single iteration of `scanned_apply_deltas`.

    Parameters
    ----------
    deltas : jax.Array
        Displacement vectors.
    pooled_labels : jax.Array
        Pooled labels carried over from the previous iteration.
    kernel_size : Sequence[int]
        Kernel size or window size of the sum pooling operation.

    Returns
    -------
    pooled_labels : jax.Array
        Pooled labels to carry to the next iteration.
    None
    """

    pooled_labels = sum_pool(deltas, pooled_labels, kernel_size)

    return pooled_labels, None


def sum_pool(
        deltas: jax.Array,
        labels: jax.Array,
        kernel_size: Sequence[int] = (3, 3)
) -> jax.Array:

    """Sum pool labels using deltas.

    Parameters
    ----------
    deltas : jax.Array
        Displacement vectors.
    labels : jax.Array
        Binary labels.
    kernel_size : Sequence[int], optional
        Kernel size or window size of the sum pooling operation. Default is (3, 3).

    Returns
    -------
    pooled_labels : jax.Array
        Pooled labels.
    """

    # Generate an index map.
    i, j = jnp.arange(deltas.shape[0]), jnp.arange(deltas.shape[1])
    index_map = jnp.stack(jnp.meshgrid(i, j, indexing='ij'), axis=-1)

    # Compute the pixel convergence array after applying deltas.
    convergence = jnp.rint(deltas + index_map).astype(int)

    # Pad convergence and labels arrays.
    pad_width = (((kernel_size[0] - 1) // 2, ) * 2, ((kernel_size[1] - 1) // 2, ) * 2)
    convergence = jnp.pad(convergence, (*pad_width, (0, 0)))
    labels = jnp.pad(labels, pad_width)

    # Sum pool the label values of pixels that converge at each pixel location.
    pooled_labels = vmap_sum_convergent_labels(convergence, labels, kernel_size, i, j)

    return pooled_labels


vmap_sum_pool = vmap(sum_pool, in_axes=(0, 0, None))


def _sum_convergent_labels(
        convergence: jax.Array,
        labels: jax.Array,
        kernel_size: Sequence[int],
        i: jax.Array,
        j: jax.Array
) -> jax.Array:

    """Sum pool the label values of pixels that converge at a given pixel location.

    Parameters
    ----------
    convergence : jax.Array
        Convergence array.
    labels : jax.Array
        Binary labels.
    kernel_size : Sequence[int]
        Kernel size or window size to search for labels convergence.
    i : jax.Array
        Pixel row index.
    j : jax.Array
        Pixel column index.

    Returns
    -------
    pooled_label : jax.Array
        Pooled label.
    """

    # Extract the convergence and labels arrays in the kernel.
    convergence = dynamic_slice(convergence, (i, j, 0), (*kernel_size, 2))
    labels = dynamic_slice(labels, (i, j), kernel_size)

    # Find pixel sources that converge at the given pixel location.
    sources = jnp.all(convergence == jnp.array((i, j)), axis=-1)

    # Sum pool the label values at pixel sources.
    pooled_label = jnp.sum(sources * labels)

    return pooled_label


vmap_sum_convergent_labels = vmap(
    vmap(_sum_convergent_labels, in_axes=(None, None, None, None, 0)),
    in_axes=(None, None, None, 0, None)
)


def smooth_sum_pool(
        deltas: jax.Array,
        labels: jax.Array,
        sigma: float = 0.5,
        kernel_size: Sequence[int] = (3, 3),
        epsilon: float = 1e-7
) -> jax.Array:

    """Sum pool labels using deltas.

    Unlike conventional pooling, which takes a discrete summation, each pixel's contribution to the pooled result of
    neighboring pixels is determined by an isotropic Gaussian distribution. This Gaussian is centered at the pixel's
    convergence coordinates given by `delta` and has standard deviation `sigma` in each axis. This operation is
    designed to be differentiable and thus suitable to be used within a loss function during training.

    Parameters
    ----------
    deltas : jax.Array
        Displacement vectors.
    labels : jax.Array
        Binary labels.
    sigma : jax.Array
        Standard deviation of the Gaussian distribution. Default is 0.5.
    kernel_size : Sequence[int], optional
        Kernel size or window size of the sum pooling operation. Default is (3, 3).
    epsilon : float, optional
        Small constant for numerical stability. Default is 1e-7.

    Returns
    -------
    pooled_labels : jax.Array
        Pooled labels.
    """

    # Generate an index map.
    i, j = jnp.arange(deltas.shape[0]), jnp.arange(deltas.shape[1])
    index_map = jnp.stack(jnp.meshgrid(i, j, indexing='ij'), axis=-1)

    # Compute the pixel convergence array after applying deltas.
    convergence = deltas + index_map

    # Pad convergence and labels arrays.
    pad_width = (((kernel_size[0] - 1) // 2, ) * 2, ((kernel_size[1] - 1) // 2, ) * 2)
    index_map = jnp.pad(index_map, (*pad_width, (0, 0)))
    labels = jnp.pad(labels, pad_width)

    # Compute Gaussian distributions.
    gaussians = vmap_compute_gaussian_distributions(index_map, convergence, sigma, kernel_size, epsilon, i, j)
    gaussians = jnp.pad(gaussians, (*pad_width, (0, 0), (0, 0)))

    # Distribute labels.
    pooled_labels = vmap_distribute_labels(gaussians, labels, kernel_size, i, j)

    return pooled_labels


vmap_smooth_sum_pool = vmap(smooth_sum_pool, in_axes=(0, 0, None, None, None))


def _compute_gaussian_distributions(
        index_map: jax.Array,
        convergence: jax.Array,
        sigma: float,
        kernel_size: Sequence[int],
        epsilon: float,
        i: jax.Array,
        j: jax.Array
) -> jax.Array:

    """Compute a Gaussian distribution centered at the convergence coordinates of a given pixel location.

    Parameters
    ----------
    index_map : jax.Array
        Index map.
    convergence : jax.Array
        Convergence array.
    sigma : float
        Standard deviation of the Gaussian distribution.
    kernel_size : Sequence[int]
        Kernel size or window size of the Gaussian distribution.
    epsilon : float
        Small constant for numerical stability.
    i : jax.Array
        Pixel row index.
    j : jax.Array
        Pixel column index.

    Returns
    -------
    gaussian : jax.Array
        Gaussian distribution centered at the convergence coordinates of the given pixel location.
    """

    # Extract the index map in the kernel.
    index_map = dynamic_slice(index_map, (i, j, 0), (*kernel_size, 2))

    # Compute the Gaussian distribution centered at the convergence coordinates.
    gaussian = jnp.exp(-jnp.sum((index_map - convergence) ** 2, axis=-1) / (2 * sigma ** 2))
    gaussian = gaussian / (jnp.sum(gaussian) + epsilon)

    return gaussian


vmap_compute_gaussian_distributions = vmap(
    vmap(_compute_gaussian_distributions, in_axes=(None, 0, None, None, None, None, 0)),
    in_axes=(None, 0, None, None, None, 0, None)
)


def _distribute_labels(
        distributions: jax.Array,
        labels: jax.Array,
        kernel_size: Sequence[int],
        i: jax.Array,
        j: jax.Array
) -> jax.Array:

    """Distribute the label values of nearby pixels according to a given set of distributions.

    Parameters
    ----------
    distributions : jax.Array
        Distributions array.
    labels : jax.Array
        Binary labels.
    kernel_size : Sequence[int]
        Kernel size or window size to distribute labels.
    i : jax.Array
        Pixel row index.
    j : jax.Array
        Pixel column index.

    Returns
    -------
    pooled_label : jax.Array
        Pooled label.
    """

    # Extract the distributions and labels arrays in the kernel.
    distributions = dynamic_slice(distributions, (i, j, 0, 0), (*kernel_size, *kernel_size))
    labels = dynamic_slice(labels, (i, j), kernel_size)

    # Extract weights from the distributions array.
    k, m = np.indices(kernel_size)
    weights = distributions[k, m, kernel_size[0] - 1 - k, kernel_size[1] - 1 - m]

    # Distribute the label values of nearby pixels.
    pooled_label = jnp.sum(weights * labels)

    return pooled_label


vmap_distribute_labels = vmap(
    vmap(_distribute_labels, in_axes=(None, None, None, None, 0)),
    in_axes=(None, None, None, 0, None)
)


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
    padded_size = (max(image.shape[0] for image in images), max(image.shape[1] for image in images))

    # Pad images.
    padded_images = []
    for image in images:
        pad_i = padded_size[0] - image.shape[0]
        pad_j = padded_size[1] - image.shape[1]
        pad_width = ((0, pad_i), (0, pad_j))
        if (pad_i > 0) or (pad_j > 0):
            padded_image = np.pad(image, pad_width)
        else:
            padded_image = image
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
