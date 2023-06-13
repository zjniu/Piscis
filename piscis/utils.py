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
        counts: np.ndarray,
        threshold: float,
        min_distance: int,
) -> np.ndarray:

    """Compute spot coordinates from deltas and labels.

    Parameters
    ----------
    deltas : np.ndarray
        Subpixel displacements.
    counts : np.ndarray
        Mass landscape.
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
    stack = counts.ndim == 3

    if stack:

        # Use connected components to detect spots if labels are in a stack.
        labels = measure.label(counts > threshold)
        peaks = np.array([region['centroid'] for region in measure.regionprops(labels)], dtype=int)

    else:

        # Use peak local maxima to detect spots if labels are not in a stack.
        peaks = feature.peak_local_max(counts, min_distance=min_distance, threshold_abs=threshold, exclude_border=False)

    # Apply deltas to detected spots.
    if len(peaks) > 0:
        if stack:
            coords = peaks + np.pad(deltas[peaks[:, 0], peaks[:, 1], peaks[:, 2]], ((0, 0), (1, 0)))
        else:
            coords = peaks + deltas[peaks[:, 0], peaks[:, 1]]
    else:
        coords = np.empty((0, counts.ndim), dtype=np.float32)

    return coords


def scanned_apply_deltas(
        deltas: jnp.ndarray,
        labels: jnp.ndarray,
        n_iter: int,
        kernel_size: Sequence[int] = (5, 5)
) -> jnp.ndarray:

    """Scanned version of `apply_deltas`.

    Parameters
    ----------
    deltas : jnp.ndarray
        Subpixel displacements.
    labels : jnp.ndarray
        Binary labels.
    n_iter : int
        Number of iterations.
    kernel_size : Sequence[int], optional
        Kernel size or size of the window to search for labels convergence. Default is (5, 5).

    Returns
    -------
    counts : jnp.ndarray
        Mass landscape of labels after applying deltas for `n_iter` iterations.
    """

    carry, _ = scan(lambda c, x: _scanned_apply_deltas(c, kernel_size), (deltas, labels), jnp.empty(n_iter))
    counts = carry[1]

    return counts


vmap_scanned_apply_deltas = vmap(scanned_apply_deltas, in_axes=(0, 0, None, None))


def _scanned_apply_deltas(
        carry: Tuple[jnp.ndarray, jnp.ndarray],
        kernel_size: Sequence[int]
) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:

    """Single iteration of `scanned_apply_deltas`.

    Parameters
    ----------
    carry : Tuple[jnp.ndarray, jnp.ndarray]
        Variables carried over from the previous iteration.
    kernel_size : Sequence[int]
        Kernel size or size of the window to search for labels convergence.

    Returns
    -------
    carry : Tuple[jnp.ndarray, jnp.ndarray]
        Variables to carry to the next iteration.
    counts : jnp.ndarray
        Mass landscape of labels after applying deltas for one iteration.
    """

    deltas, counts = carry
    counts = apply_deltas(deltas, counts, kernel_size)
    carry = deltas, counts

    return carry, counts


def apply_deltas(
        deltas: jnp.ndarray,
        labels: jnp.ndarray,
        kernel_size: Sequence[int] = (3, 3)
) -> jnp.ndarray:

    """Apply deltas to labels.

    Parameters
    ----------
    deltas : jnp.ndarray
        Subpixel displacements.
    labels : jnp.ndarray
        Binary labels.
    kernel_size : Sequence[int], optional
        Kernel size or size of the window to search for labels convergence. Default is (3, 3).

    Returns
    -------
    counts : jnp.ndarray
        Mass landscape of labels after applying deltas.
    """

    # Generate index map.
    i, j = jnp.arange(deltas.shape[0]), jnp.arange(deltas.shape[1])
    ii, jj = jnp.meshgrid(i, j, indexing='ij')
    index_map = jnp.stack((ii, jj), axis=-1)

    # Compute the pixel convergence array after applying deltas.
    convergence = deltas + index_map

    # Pad convergence and labels arrays.
    pad = (((kernel_size[0] - 1) // 2, ) * 2, ((kernel_size[1] - 1) // 2, ) * 2)
    convergence = jnp.pad(convergence, (*pad, (0, 0)))
    labels = jnp.pad(labels, pad)

    # Vectorize the counts convergence function.
    vmap_count_convergence = vmap(vmap(_count_convergence,
                                       in_axes=(None, None, None, None, 0)), in_axes=(None, None, None, 0, None))

    # Count the number of pixels that converge at each pixel location.
    counts = vmap_count_convergence(convergence, labels, kernel_size, i, j)

    return counts


vmap_apply_deltas = vmap(apply_deltas, in_axes=(0, 0, None))


def _count_convergence(
        convergence: jnp.ndarray,
        labels: jnp.ndarray,
        kernel_size: Sequence[int],
        i: jnp.ndarray,
        j: jnp.ndarray
) -> jnp.ndarray:

    """Count the number of pixels that converge at a given pixel location.

    Parameters
    ----------
    convergence : jnp.ndarray
        Convergence array.
    labels : jnp.ndarray
        Binary labels.
    kernel_size : Sequence[int]
        Kernel size or size of the window to search for labels convergence.
    i : jnp.ndarray
        Pixel row index.
    j : jnp.ndarray
        Pixel column index.

    Returns
    -------
    count : jnp.ndarray
        Number of pixels that converge at the given pixel location.
    """

    # Extract the convergence and labels arrays in the kernel.
    convergence = dynamic_slice(convergence, (i, j, 0), (*kernel_size, 2))
    labels = dynamic_slice(labels, (i, j), kernel_size)

    # Search for pixel sources that converge at the given pixel location.
    sources = _search_convergence(convergence, i, j)

    # Sum values of labels at pixel sources.
    count = jnp.sum(sources * labels)

    return count


def _search_convergence(
        convergence: jnp.ndarray,
        i: jnp.ndarray,
        j: jnp.ndarray
) -> jnp.ndarray:

    """Search for pixel sources that converge at a given pixel location.

    Parameters
    ----------
    convergence : jnp.ndarray
        Convergence array.
    i : jnp.ndarray
        Pixel row index.
    j : jnp.ndarray
        Pixel column index.

    Returns
    -------
    sources : jnp.ndarray
        Pixel sources that converge at the given pixel location.
    """

    sources = (i - 0.5 < convergence[:, :, 0]) & (convergence[:, :, 0] < i + 0.5) & \
              (j - 0.5 < convergence[:, :, 1]) & (convergence[:, :, 1] < j + 0.5)

    return sources


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
        raise ValueError('window_size must be an odd integer.')

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
        window_size: int = 3
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
        raise ValueError('window_size must be an odd integer.')

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

    for coord in coords:

        # Crop the image around the coordinate.
        index = np.rint(coord).astype(int)
        cropped_image = image[index[0]: index[0] + window_size, index[1]: index[1] + window_size]

        # Fit a Gaussian to the cropped image.
        try:
            popt, pcov = optimize.curve_fit(_gaussian, (x, y), cropped_image.ravel(),
                                            p0=(np.max(cropped_image), 0, 0, 1, 1, 0, np.min(cropped_image)),
                                            bounds=((np.max(cropped_image) / 2, -delta, -delta, 0, 0, 0, 0),
                                                    (2 * np.max(cropped_image), delta, delta, np.inf, np.inf,
                                                     2 * np.pi, np.inf)))
            fitted_coords.append(index + np.array([popt[2], popt[1]]))
        except RuntimeError:
            pass

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
