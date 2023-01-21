import jax.numpy as np
import numpy as onp

from jax import vmap
from jax.lax import dynamic_slice
from skimage import feature, measure


def compute_spot_coordinates(deltas, counts, threshold, min_distance):

    stack = counts.ndim == 3
    counts = onp.asarray(counts)

    if stack:
        labels = measure.label(counts > threshold)
        peaks = onp.array([region['centroid'] for region in measure.regionprops(labels)], dtype=int)
    else:
        peaks = feature.peak_local_max(counts, min_distance=min_distance, threshold_abs=threshold, exclude_border=False)

    if len(peaks) > 0:
        if stack:
            coords = peaks + onp.pad(onp.asarray(deltas)[peaks[:, 0], peaks[:, 1], peaks[:, 2]], ((0, 0), (1, 0)))
        else:
            coords = peaks + onp.asarray(deltas)[peaks[:, 0], peaks[:, 1]]
    else:
        coords = onp.empty((0, counts.ndim), dtype=onp.float32)

    return coords


def colocalize_pixels(deltas, labels):

    i, j = np.arange(deltas.shape[0]), np.arange(deltas.shape[1])
    ii, jj = np.meshgrid(i, j, indexing='ij')
    index_map = np.stack((ii, jj), axis=-1)

    convergence = deltas + index_map
    convergence = np.pad(convergence, ((1, 1), (1, 1), (0, 0)))
    labels = np.pad(labels, ((1, 1), (1, 1)))

    vmap_count_convergence = vmap(vmap(_count_convergence,
                                       in_axes=(None, None, None, 0)), in_axes=(None, None, 0, None))
    counts = vmap_count_convergence(convergence, labels, i, j)

    return counts


vmap_colocalize_pixels = vmap(colocalize_pixels, in_axes=(0, 0))


def _count_convergence(convergence, labels, i, j):

    convergence = dynamic_slice(convergence, (i, j, 0), (3, 3, 2))
    labels = dynamic_slice(labels, (i, j), (3, 3))
    sources = _search_convergence(convergence, i, j)
    count = np.sum(sources * labels)

    return count


def _search_convergence(convergence, i, j):

    sources = (i - 0.5 < convergence[:, :, 0]) & (convergence[:, :, 0] < i + 0.5) \
              & (j - 0.5 < convergence[:, :, 1]) & (convergence[:, :, 1] < j + 0.5)

    return sources


def match_coords(coords, i, set_id, coord_set_ids, checked, threshold):

    distances = onp.sqrt(onp.sum((coords[i] - coords) ** 2, axis=1))
    matches = list(onp.where(distances < threshold)[0])
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
        new_coords.append(onp.mean(old_coords, axis=0))

    new_coords = onp.array(new_coords)

    return new_coords
