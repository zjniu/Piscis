import jax.numpy as jnp
import numpy as np

from jax import vmap
from jax.lax import dynamic_slice, scan
from skimage import feature, measure


def compute_spot_coordinates(deltas, labels, threshold, min_distance):

    stack = labels.ndim == 3

    if stack:
        labels = measure.label(labels > threshold)
        peaks = np.array([region['centroid'] for region in measure.regionprops(labels)], dtype=int)
    else:
        peaks = feature.peak_local_max(labels, min_distance=min_distance, threshold_abs=threshold, exclude_border=False)

    if len(peaks) > 0:
        if stack:
            coords = peaks + np.pad(deltas[peaks[:, 0], peaks[:, 1], peaks[:, 2]], ((0, 0), (1, 0)))
        else:
            coords = peaks + deltas[peaks[:, 0], peaks[:, 1]]
    else:
        coords = np.empty((0, labels.ndim), dtype=np.float32)

    return coords


def scanned_colocalize_pixels(deltas, labels, n_iter, kernel_size=(5, 5)):

    carry, _ = scan(lambda c, x: _scanned_colocalize_pixels(c, x, kernel_size), (deltas, labels), jnp.empty(n_iter))
    counts = carry[1]

    return counts


vmap_scanned_colocalize_pixels = vmap(scanned_colocalize_pixels, in_axes=(0, 0))


def _scanned_colocalize_pixels(carry, x, kernel_size):

    deltas, counts = carry
    counts = colocalize_pixels(deltas, counts, kernel_size)
    carry = deltas, counts

    return carry, counts


def colocalize_pixels(deltas, labels, kernel_size=(3, 3)):

    i, j = jnp.arange(deltas.shape[0]), jnp.arange(deltas.shape[1])
    ii, jj = jnp.meshgrid(i, j, indexing='ij')
    index_map = jnp.stack((ii, jj), axis=-1)

    convergence = deltas + index_map

    pad = (((kernel_size[0] - 1) // 2, ) * 2, ((kernel_size[1] - 1) // 2, ) * 2)
    convergence = jnp.pad(convergence, (*pad, (0, 0)))
    labels = jnp.pad(labels, pad)

    vmap_count_convergence = vmap(vmap(_count_convergence,
                                       in_axes=(None, None, None, None, 0)), in_axes=(None, None, None, 0, None))
    counts = vmap_count_convergence(convergence, labels, kernel_size, i, j)

    return counts


vmap_colocalize_pixels = vmap(colocalize_pixels, in_axes=(0, 0, None))


def _count_convergence(convergence, labels, kernel_size, i, j):

    convergence = dynamic_slice(convergence, (i, j, 0), (*kernel_size, 2))
    labels = dynamic_slice(labels, (i, j), kernel_size)
    sources = _search_convergence(convergence, i, j)
    count = jnp.sum(sources * labels)

    return count


def _search_convergence(convergence, i, j):

    sources = (i - 0.5 < convergence[:, :, 0]) & (convergence[:, :, 0] < i + 0.5) \
              & (j - 0.5 < convergence[:, :, 1]) & (convergence[:, :, 1] < j + 0.5)

    return sources


def match_coords(coords, i, set_id, coord_set_ids, checked, threshold):

    distances = np.sqrt(np.sum((coords[i] - coords) ** 2, axis=1))
    matches = list(np.where(distances < threshold)[0])
    matches.remove(i)
    for match in matches:
        coord_set_ids[match] = set_id
    checked.append(i)

    return matches


def remove_duplicate_coords(coords, threshold=1):

    sets = {}
    coord_set_ids = {i: None for i in range(len(coords))}
    checked = []

    for i, set_id in coord_set_ids.items():

        if set_id is None:

            set_id = len(sets)
            coord_set_ids[i] = set_id
            new_set = [i]
            sets[set_id] = new_set

            matches = match_coords(coords, i, set_id, coord_set_ids, checked, threshold)
            new_set.extend(matches)
            to_check = matches.copy()

            while len(to_check) > 0:

                for j in to_check:

                    matches = match_coords(coords, j, set_id, coord_set_ids, checked, threshold)
                    to_check.remove(j)
                    new_matches = set(matches) - set(new_set)
                    new_set.extend(new_matches)
                    to_check.extend(new_matches)

                to_check = list(set(to_check) - set(checked))

    new_coords = []
    for s in sets.values():

        old_coords = coords[list(s)]
        new_coords.append(np.mean(old_coords, axis=0))

    new_coords = np.array(new_coords)

    return new_coords
