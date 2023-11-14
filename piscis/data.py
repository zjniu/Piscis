import deeptile
import jax
import numpy as np

from jax import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from piscis.utils import pad, remove_duplicate_coords
from piscis.transforms import batch_adjust, RandomAugment, voronoi_transform


def generate_dataset(
        path: str,
        images: List[np.ndarray],
        coords: List[np.ndarray],
        key: jax.Array,
        tile_size: Tuple[int, int] = (256, 256),
        min_spots: int = 1,
        train_size: float = 0.70,
        test_size: float = 0.15
) -> None:

    """Generate a dataset from images and spot coordinates.

    Parameters
    ----------
    path : str
        Path to save dataset.
    images : List[np.ndarray]
        List of images.
    coords : List[np.ndarray]
        List of ground truth spot coordinates.
    key : jax.Array
        Random key used for splitting the dataset into training, validation, and test sets.
    tile_size : Tuple[int, int], optional
        Tile size used for splitting images. Default is (256, 256).
    min_spots : int, optional
        Minimum number of spots per tile. Default is 1.
    train_size : float, optional
        Fraction of dataset used for training. Default is 0.70.
    test_size : float, optional
        Fraction of dataset used for testing. Default is 0.15.
    """

    # Remove duplicate coordinates.
    for i in range(len(coords)):
        coords[i] = remove_duplicate_coords(coords[i])

    # Create a DeepTile object and get tiles.
    images = np.stack(images)
    dt = deeptile.load(images)
    tiles1 = dt.get_tiles(tile_size, (0, 0))
    tiles2 = tiles1.import_data(coords, 'coords')
    nonempty_tiles = [(image.compute(), coord)
                      for tile1, tile2 in zip(tiles1[tiles1.nonempty_indices], tiles2[tiles2.nonempty_indices])
                      for image, coord in zip(tile1, tile2) if len(coord) > min_spots]
    tiled_images_list, tiled_coords_list = zip(*nonempty_tiles)
    tiled_images = np.empty(len(tiled_images_list), dtype=object)
    tiled_coords = np.empty(len(tiled_coords_list), dtype=object)
    tiled_images[:] = tiled_images_list
    tiled_coords[:] = tiled_coords_list

    # Randomly shuffle the tiles.
    size = len(tiled_images)
    perms = np.asarray(random.permutation(key, size))
    tiled_images = tiled_images[perms]
    tiled_coords = tiled_coords[perms]

    # Split the dataset into training, validation, and test sets.
    split_indices = np.rint(np.cumsum((train_size, test_size)) * size).astype(int)
    x_train = tiled_images[:split_indices[0]]
    y_train = tiled_coords[:split_indices[0]]
    x_valid = tiled_images[split_indices[1]:]
    y_valid = tiled_coords[split_indices[1]:]
    x_test = tiled_images[split_indices[0]:split_indices[1]]
    y_test = tiled_coords[split_indices[0]:split_indices[1]]

    # Create the dataset dictionary.
    np.savez(path, x_train=x_train, y_train=y_train, x_valid=x_valid, y_valid=y_valid, x_test=x_test, y_test=y_test)


def load_datasets(
        path: str,
        adjustment: Optional[str] = 'standardize',
        load_train: bool = True,
        load_valid: bool = True,
        load_test: bool = True
) -> Dict:

    """Load datasets from a directory or file.

    Parameters
    ----------
    path : str
        Path to a dataset or directory of datasets.
    adjustment : Optional[str], optional
        Adjustment type applied to images. Supported types are 'normalize' and 'standardize'. Default is 'standardize'.
    load_train : bool, optional
        Whether to load the training set. Default is True.
    load_valid : bool, optional
        Whether to load the validation set. Default is True.
    load_test : bool, optional
        Whether to load the test set. Default is True.

    Returns
    -------
    dataset : Dict
        Dataset dictionary.
    """

    # Create empty dictionaries.
    train = {'images': [], 'coords': []}
    valid = {'images': [], 'coords': []}
    test = {'images': [], 'coords': []}
    dataset = {}

    # Get dataset paths.
    path = Path(path)
    if path.is_file() and path.suffix == '.npz':
        dataset_paths = [path]
    else:
        dataset_paths = path.glob('*.npz')

    # Load datasets.
    for dataset_path in dataset_paths:
        npz = np.load(dataset_path, allow_pickle=True)
        if load_train:
            train['images'].append(npz['x_train'])
            train['coords'].append(npz['y_train'])
        if load_valid:
            valid['images'].append(npz['x_valid'])
            valid['coords'].append(npz['y_valid'])
        if load_test:
            test['images'].append(npz['x_test'])
            test['coords'].append(npz['y_test'])

    # Combine datasets and adjust images if necessary.
    if load_train:
        train['images'] = np.concatenate(train['images'])
        train['images'] = batch_adjust(train['images'], adjustment)
        train['coords'] = np.concatenate(train['coords'])
        dataset['train'] = train
    if load_valid:
        valid['images'] = np.concatenate(valid['images'])
        valid['images'] = batch_adjust(valid['images'], adjustment)
        valid['coords'] = np.concatenate(valid['coords'])
        dataset['valid'] = valid
    if load_test:
        test['images'] = np.concatenate(test['images'])
        test['images'] = batch_adjust(test['images'], adjustment)
        test['coords'] = np.concatenate(test['coords'])
        dataset['test'] = test

    return dataset


def transform_subdataset(
        subds: Dict,
        input_size: Tuple[int, int],
        min_spots: int = 1,
        key: Optional[jax.Array] = None
) -> Dict:

    """Transform subdataset for model training and validation.

    Parameters
    ----------
    subds : Dict
        Subdataset dictionary.
    input_size : Tuple[int, int]
        Size of the input images used during training.
    min_spots : int, optional
        Minimum number of spots per image. Default is 1.
    key : Optional[jax.Array], optional
        Random key used for data augmentation. Default is None.

    Returns
    -------
    transformed_subds : Dict
        Transformed subdataset dictionary.
    """

    if key is not None:

        base_scales = np.ones(len(subds['images']))

        # Create random augmentation transformer.
        transformer = RandomAugment()
        transformer.generate_transforms(subds['images'], key, base_scales, input_size)

        # Apply transformations.
        images = transformer.apply_image_transforms(subds['images'], interpolation='bilinear')
        coords_list = transformer.apply_coord_transforms(subds['coords'], filter_coords=True)

    else:

        images = pad(subds['images'])
        coords_list = list(subds['coords'])

    # Remove images with less than min_spots.
    counts = np.array([len(coord) for coord in coords_list])
    pop_indices = np.where((counts < min_spots))[0]
    for index in np.flip(pop_indices):
        images.pop(index)
        coords_list.pop(index)
    coords = np.empty(len(coords_list), dtype=object)
    coords[:] = coords_list

    # Created the transformed subdataset dictionary.
    transformed_subds = {
        'images': np.array(images)[:, :, :, None],
        'coords': coords,
    }

    return transformed_subds


def transform_batch(
        batch: Dict,
        dilation_iterations: int = 1,
        coords_pad_length: Optional[int] = None
):

    """Transform batch for model training and validation.

    Parameters
    ----------
    batch : Dict
        Batch dictionary.
    dilation_iterations : int, optional
        Number of iterations to dilate ground truth labels. Default is 1.
    coords_pad_length : Optional[int], optional
        Padded length of the coordinates sequence. Default is None.
    """

    images = batch['images']
    coords = batch['coords']
    output_shape = images.shape[1:3]

    # Apply the Voronoi transform.
    deltas, labels = voronoi_transform(coords, output_shape, dilation_iterations, coords_pad_length)

    # Create the transformed batch dictionary.
    transformed_batch = {
        'images': images,
        'deltas': deltas,
        'labels': labels
    }

    return transformed_batch
