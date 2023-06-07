import cv2 as cv
import jax.numpy as jnp
import numpy as np

from jax import jit, random, vmap
from scipy import ndimage


class RandomAugment:

    def __init__(self):

        self.output_shape = None
        self.intensity_scales = []
        self.scales = []
        self.dxys = []
        self.thetas = []
        self.affines = []
        self.flips0 = []
        self.flips1 = []

    def generate_transforms(self, images, key, base_scales, output_shape, max_intensity_scale_factor=5,
                            min_scale_factor=0.75, max_scale_factor=1.25):

        # Reset
        self.output_shape = output_shape
        self.flips0 = []
        self.flips1 = []
        self.scales = []
        self.dxys = []
        self.thetas = []
        self.affines = []
        self.intensity_scales = []

        for image, base_scale in zip(images, base_scales):

            # Random flip
            key, *subkeys = random.split(key, 3)
            flip0 = random.uniform(subkeys[0]) > 0.5
            self.flips0.append(float(flip0))
            flip1 = random.uniform(subkeys[1]) > 0.5
            self.flips1.append(float(flip1))

            # Random scaling
            key, subkey = random.split(key)
            scale = base_scale * (min_scale_factor + (max_scale_factor - min_scale_factor) * random.uniform(subkey))
            self.scales.append(float(scale))

            # Random translation
            key, subkey = random.split(key)
            dxy = np.maximum(0, np.array([image.shape[1] * scale - output_shape[1],
                                          image.shape[0] * scale - output_shape[0]]))
            dxy = (random.uniform(subkey, (2,)) - 0.5) * dxy
            self.dxys.append(np.asarray(dxy))

            # Random rotation
            key, subkey = random.split(key)
            theta = random.uniform(subkey) * 2 * np.pi
            self.thetas.append(float(theta))

            # Construct affine transformation
            image_center = (image.shape[1] / 2, image.shape[0] / 2)
            affine = cv.getRotationMatrix2D(image_center, float(theta * 180 / np.pi), float(scale))
            affine[:, 2] += np.array(output_shape) / 2 - np.array(image_center) + dxy
            self.affines.append(affine)

            # Random intensity scaling
            key, subkey = random.split(key)
            intensity_scale = jnp.exp((random.uniform(subkey) - 0.5) * 2 * jnp.log(max_intensity_scale_factor))
            self.intensity_scales.append(float(intensity_scale))

    def apply_coord_transforms(self, coords, filter_coords=True):

        transformed_coords = []

        for coord, flip0, flip1, affine in zip(coords, self.flips0, self.flips1, self.affines):

            # Apply affine transformation
            coord = np.concatenate((np.flip(coord, axis=1), np.ones((len(coord), 1))), axis=1)
            transformed_coord = np.flip((coord @ affine.T), axis=1)

            # Random flip
            if flip0:
                transformed_coord[:, 0] = self.output_shape[0] - 1 - transformed_coord[:, 0]
            if flip1:
                transformed_coord[:, 1] = self.output_shape[1] - 1 - transformed_coord[:, 1]

            # Filter coordinates outside transformed image
            if filter_coords:
                transformed_coord = transformed_coord[np.all((transformed_coord > -0.5) &
                                                             (transformed_coord < np.array(self.output_shape) - 0.5),
                                                             axis=1)]

            transformed_coords.append(transformed_coord)

        return transformed_coords

    def apply_image_transforms(self, images, interpolation='nearest'):

        transformed_images = []

        for image, affine, flip0, flip1, intensity_scale in \
                zip(images, self.affines, self.flips0, self.flips1, self.intensity_scales):

            # Apply affine transformation
            if interpolation == 'nearest':
                image = cv.warpAffine(image, M=affine, dsize=self.output_shape, flags=cv.INTER_NEAREST)
            elif interpolation == 'bilinear':
                image = cv.warpAffine(image, M=affine, dsize=self.output_shape, flags=cv.INTER_LINEAR)
            else:
                raise ValueError('Interpolation mode not supported.')

            # Random flip
            if flip0:
                image = np.flip(image, axis=0)
            if flip1:
                image = np.flip(image, axis=1)

            # Random intensity scaling
            image = image * intensity_scale

            transformed_images.append(image)

        return transformed_images


def batch_adjust(images, adjustment, **kwargs):

    if adjustment is not None:
        images = list(images)
        adjusted_images = np.empty(len(images), dtype=object)
        for i, image in enumerate(images):
            adjusted_images[i] = adjust(image, adjustment, **kwargs)
    else:
        adjusted_images = images

    return adjusted_images


def adjust(image, adjustment, **kwargs):

    if adjustment == 'normalize':
        image = normalize(image, **kwargs)
    elif adjustment == 'standardize':
        image = standardize(image, **kwargs)
    else:
        raise ValueError('Adjustment type not supported.')

    return image


def normalize(image, lower=0, upper=100, epsilon=1e-7):

    image_lower = np.percentile(image, lower)
    image_upper = np.percentile(image, upper)

    return (image - image_lower) / (image_upper - image_lower + epsilon)


def standardize(image, epsilon=1e-7):

    return (image - np.mean(image)) / (np.std(image) + epsilon)


def subpixel_distance_transform(coords_list, coords_pad_length=None, shape=(256, 256), dy=1.0, dx=1.0):

    """For each pixel in an image, return the vertical and horizontal distances to a point in
    ``coords`` that is in the pixel nearest to it.
    """

    batch_size = len(coords_list)
    labels = np.zeros((batch_size,) + shape, dtype=bool)

    max_num_coords = np.max([len(coords) for coords in coords_list])
    if (coords_pad_length is None) or (coords_pad_length < max_num_coords):
        coords_pad_length = max_num_coords

    subpixel_coords = np.zeros((batch_size, coords_pad_length, 2))
    rounded_coords = np.zeros((batch_size, coords_pad_length, 2), dtype=int)
    edt_indices = np.zeros((batch_size, 2) + shape, dtype=int)

    for i, coords in enumerate(coords_list):

        coords = coords[(coords[:, 0] >= -0.5) & (coords[:, 0] <= shape[0] - 0.5) &
                        (coords[:, 1] >= -0.5) & (coords[:, 1] <= shape[1] - 0.5)]

        padding = ((0, coords_pad_length - len(coords)), (0, 0))
        subpixel_coords[i] = np.pad(coords, padding, constant_values=-1)
        unpadded_rounded_coords = np.rint(coords).astype(int)
        labels[i][unpadded_rounded_coords[:, 0], unpadded_rounded_coords[:, 1]] = True
        rounded_coords[i] = np.pad(unpadded_rounded_coords, padding, constant_values=-1)
        edt_indices[i] = ndimage.distance_transform_edt(~labels[i], return_distances=False, return_indices=True,
                                                        sampling=(dy, dx))

    labels = jnp.expand_dims(labels, -1)
    subpixel_coords = jnp.array(subpixel_coords)
    rounded_coords = jnp.array(rounded_coords)
    edt_indices = jnp.array(edt_indices)

    inputs = [subpixel_coords, rounded_coords, edt_indices, dy, dx]
    vmap_vmap_sdt = vmap(_vmap_sdt, in_axes=([0, 0, 0, None, None], None, None))
    deltas, nearest = vmap_vmap_sdt(inputs, jnp.arange(shape[0]), jnp.arange(shape[1]))

    return deltas, labels, nearest


def _vmap_sdt(inputs, i, j):

    return vmap(vmap(_sdt, in_axes=(None, None, 0)), in_axes=(None, 0, None))(inputs, i, j)


@jit
def _sdt(inputs, i, j):

    subpixel_coords, rounded_coords, edt_indices, dy, dx = inputs
    nearest = jnp.where(jnp.all(rounded_coords == edt_indices[:, i, j], axis=1)[:, None], subpixel_coords, 0)
    nearest = jnp.sum(nearest, axis=0)
    delta_y = dy * (nearest[0] - i)
    delta_x = dx * (nearest[1] - j)
    delta = jnp.stack((delta_y, delta_x), axis=-1)

    return delta, nearest
