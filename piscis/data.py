import deeptile
import numpy as np

from jax import random
from pathlib import Path
from scipy import ndimage

from piscis.utils import remove_duplicate_coords
from piscis.transforms import batch_normalize, batch_standardize, RandomAugment, subpixel_distance_transform


def generate_dataset(images_list, coords_list, key, adjustment=None,
                     tile_size=(256, 256), overlap=(0, 0), min_spots=1,
                     train_size=0.70, test_size=0.15):

    for i in range(len(coords_list)):
        coords_list[i] = remove_duplicate_coords(coords_list[i])

    if adjustment == 'normalize':
        images_list = batch_normalize(images_list)
    elif adjustment == 'standardize':
        images_list = batch_standardize(images_list)
    images_list = np.stack(images_list)

    dt = deeptile.load(images_list)
    tiles1 = dt.get_tiles(tile_size, overlap)
    tiles2 = tiles1.import_data(coords_list, 'coords')
    nonempty_tiles = [(image.compute(), coords)
                      for tile1, tile2 in zip(tiles1[tiles1.nonempty_indices], tiles2[tiles2.nonempty_indices])
                      for image, coords in zip(tile1, tile2) if len(coords) > min_spots]
    tiled_images, tiled_coords = zip(*nonempty_tiles)
    tiled_images = np.array(tiled_images, dtype=object)
    tiled_coords = np.array(tiled_coords, dtype=object)

    size = len(tiled_images)
    perms = np.asarray(random.permutation(key, size))
    tiled_images = tiled_images[perms]
    tiled_coords = tiled_coords[perms]

    split_indices = np.rint(np.cumsum((train_size, test_size)) * size).astype(int)
    x_train = tiled_images[:split_indices[0]]
    y_train = tiled_coords[:split_indices[0]]
    x_valid = tiled_images[split_indices[1]:]
    y_valid = tiled_coords[split_indices[1]:]
    x_test = tiled_images[split_indices[0]:split_indices[1]]
    y_test = tiled_coords[split_indices[0]:split_indices[1]]

    dataset = {
        'x_train': x_train,
        'y_train': y_train,
        'x_valid': x_valid,
        'y_valid': y_valid,
        'x_test': x_test,
        'y_test': y_test
    }

    return dataset


def load_datasets(path, adjustment='standardize'):

    train = {'images': [], 'coords': []}
    valid = {'images': [], 'coords': []}
    test = {'images': [], 'coords': []}

    path = Path(path)
    if path.is_file() and path.suffix == '.npz':
        datasets = [path]
    else:
        datasets = path.glob('*.npz')

    for dataset in datasets:

        data = np.load(dataset, allow_pickle=True)

        train['images'].append(data['x_train'])
        train['coords'].append(data['y_train'])
        valid['images'].append(data['x_valid'])
        valid['coords'].append(data['y_valid'])
        test['images'].append(data['x_test'])
        test['coords'].append(data['y_test'])

    train['images'] = np.concatenate(train['images'])
    train['coords'] = np.concatenate(train['coords'])
    valid['images'] = np.concatenate(valid['images'])
    valid['coords'] = np.concatenate(valid['coords'])
    test['images'] = np.concatenate(test['images'])
    test['coords'] = np.concatenate(test['coords'])

    if adjustment == 'normalize':
        train['images'] = batch_normalize(train['images'])
        valid['images'] = batch_normalize(valid['images'])
        test['images'] = batch_normalize(test['images'])
        images = train['images'] + valid['images'] + test['images']
    elif adjustment == 'standardize':
        train['images'] = batch_standardize(train['images'])
        valid['images'] = batch_standardize(valid['images'])
        test['images'] = batch_standardize(test['images'])
        images = train['images'] + valid['images'] + test['images']
    else:
        images = list(train['images']) + list(valid['images']) + list(test['images'])

    padded_size = (max(image.shape[0] for image in images), max(image.shape[1] for image in images))
    for i, image in enumerate(train['images']):
        constant = np.min(image)
        train['images'][i] = np.pad(image, ((0, padded_size[0] - image.shape[0]), (0, padded_size[1] - image.shape[1])),
                                    mode='constant', constant_values=constant)
    for i, image in enumerate(valid['images']):
        constant = np.min(image)
        valid['images'][i] = np.pad(image, ((0, padded_size[0] - image.shape[0]), (0, padded_size[1] - image.shape[1])),
                                    mode='constant', constant_values=constant)
    for i, image in enumerate(test['images']):
        constant = np.min(image)
        test['images'][i] = np.pad(image, ((0, padded_size[0] - image.shape[0]), (0, padded_size[1] - image.shape[1])),
                                   mode='constant', constant_values=constant)

    train['images'] = np.stack(train['images'])
    valid['images'] = np.stack(valid['images'])
    test['images'] = np.stack(test['images'])

    ds = {
        'train': train,
        'valid': valid,
        'test': test
    }

    return ds


def transform_dataset(ds, input_size, min_spots=1, key=None):

    if key is not None:

        base_scales = np.ones(len(ds['images']))

        # Create transformer
        transformer = RandomAugment()
        transformer.generate_transforms(ds['images'], key, base_scales, input_size)

        # Apply transformations
        images = transformer.apply_image_transforms(ds['images'], interpolation='bilinear')
        coords_list = transformer.apply_coord_transforms(ds['coords'], filter_coords=True)

    else:

        images = ds['images'].tolist()
        coords_list = ds['coords'].tolist()

    # Remove images with less than min_spots
    counts = np.array([len(coord) for coord in coords_list])
    pop_indices = np.where((counts < min_spots))[0]
    for index in np.flip(pop_indices):
        images.pop(index)
        coords_list.pop(index)

    coords = np.empty(len(coords_list), dtype=object)
    coords[:] = coords_list

    transformed_ds = {
        'images': np.array(images)[:, :, :, None],
        'coords': coords,
    }

    return transformed_ds


def transform_batch(batch, coords_pad_length=None, dilation_iterations=1):

    output_shape = batch['images'].shape[1:3]
    coords = batch['coords']

    deltas, labels, _ = subpixel_distance_transform(coords, coords_pad_length, output_shape)
    labels = np.asarray(labels)

    if dilation_iterations > 0:
        dilated_labels = []
        structure = np.ones((3, 3, 1), dtype=bool)
        for label in labels:
            dilated_label = ndimage.binary_dilation(label, structure=structure, iterations=dilation_iterations)
            dilated_labels.append(dilated_label)
        dilated_labels = np.stack(dilated_labels)
    else:
        dilated_labels = labels

    transformed_batch = {
        'images': batch['images'],
        'deltas': deltas,
        'labels': labels,
        'dilated_labels': dilated_labels
    }

    return transformed_batch
