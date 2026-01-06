import deeptile
import numpy as np
import tifffile
import torch

from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

from piscis.utils import pad, remove_duplicate_coords
from piscis.transforms import adjust, batch_adjust, RandomAugment, voronoi_transform


class SpotsDataset(torch.utils.data.Dataset):

    """Spot detection dataset.

    Parameters
    ----------
    x : List[Path]
        List of image paths.
    y : List[Path]
        List of ground truth spot coordinates paths.
    adjustment : Optional[str], optional
        Adjustment type applied to images. Supported types are 'normalize' and 'standardize'. Default is 'standardize'.
    split : Optional[str], optional
        Dataset split. Default is None.
    """

    def __init__(
            self,
            x_paths: List[Path],
            y_paths: List[Path],
            adjustment: Optional[str] = 'standardize',
            split: Optional[str] = None
    ) -> None:

        self.x_paths = x_paths
        self.y_paths = y_paths
        self.adjustment = adjustment
        self.split = split

    def __len__(self) -> int:

        return len(self.x_paths)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:

        x = tifffile.imread(self.x_paths[index])
        y_path = self.y_paths[index]
        if y_path.stat().st_size == 0:
            y = np.empty((0, 2))
        else:
            y = np.loadtxt(y_path, delimiter=',').reshape(-1, 2)

        sample = (adjust(x, self.adjustment), y)

        return sample


class MultiSpotsDataset(torch.utils.data.Dataset):

    """Multi-spot detection dataset.

    Parameters
    ----------
    datasets : List[SpotsDataset]
        List of datasets.
    weights : List[float]
        List of dataset sampling weights.
    """

    def __init__(
            self,
            datasets: List[SpotsDataset],
            weights: List[float]
    ) -> None:

        self.datasets = datasets
        self.weights = torch.tensor(weights, dtype=torch.float)
        self.epoch_size = round(sum(len(ds) * w for ds, w in zip(datasets, weights)))
        self.split = datasets[0].split

    def __len__(self) -> int:
        
        return self.epoch_size

    def __getitem__(self, index) -> Tuple[np.ndarray, np.ndarray]:

        dataset_idx, sample_idx = index

        return self.datasets[dataset_idx][sample_idx]


class WeightedDatasetSampler(torch.utils.data.Sampler):

    """Weighted dataset sampler.

    Parameters
    ----------
    multi_dataset : MultiSpotsDataset
        Multi-spot detection dataset.
    num_samples : Optional[int], optional
        Number of samples to draw. Default is None.
    seed : Optional[int], optional
        Random seed. Default is None.
    """

    def __init__(
            self,
            multi_dataset: MultiSpotsDataset,
            num_samples: Optional[int] = None,
            seed: Optional[int] = None
    ):

        self.datasets = multi_dataset.datasets
        self.weights = multi_dataset.weights
        if num_samples is None:
            self.num_samples = len(multi_dataset)
        else:
            self.num_samples = num_samples
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)
        self.seed = seed

    def __len__(self):

        return self.num_samples

    def __iter__(self):

        for _ in range(self.num_samples):

            dataset_idx = torch.multinomial(self.weights, 1, generator=self.generator).item()
            sample_idx = torch.randint(0, len(self.datasets[dataset_idx]), (1, ), generator=self.generator).item()

            yield dataset_idx, sample_idx


class SpotsDataStream(torch.utils.data.IterableDataset):

    """Spot detection data stream.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Torch dataset.
    batch_size : int
        Batch size.
    epoch : int, optional
        Current epoch. Default is 1.
    seed : int, optional
        Random seed. Default is 0.
    shuffle : bool, optional
        Whether to shuffle the dataset. Default is True.
    augment_cls : Optional[Callable], optional
        Augmentation class. Default is None.
    """
        
    def __init__(
            self,
            dataset: torch.utils.data.Dataset,
            batch_size: int,
            epoch: int = 1,
            seed: int = 0,
            shuffle: bool = True,
            augment_cls: Optional[Callable] = None
    ) -> None:

        self.dataset = dataset
        self.batch_size = batch_size
        self.epoch = epoch
        self.seed = seed
        self.shuffle = shuffle
        self.augment_cls = augment_cls
        self.cached_indices = None

    def set_epoch(self, epoch: int) -> None:
        
        self.epoch = epoch

    def next_epoch(self) -> None:

        self.epoch += 1

    def _make_sampler(
            self,
            sampler_seed: int,
            worker_id: int,
            num_workers: int
    ) -> Union[torch.utils.data.Sampler, range]:

        if isinstance(self.dataset, (SpotsDataset, torch.utils.data.ConcatDataset)):
            if self.shuffle:
                worker_seed = sampler_seed + worker_id
                generator = torch.Generator().manual_seed(worker_seed)
                sampler = torch.utils.data.RandomSampler(self.dataset, num_samples=len(self.dataset), generator=generator)
            else:
                sampler = range(worker_id, len(self.dataset), num_workers)
        elif isinstance(self.dataset, MultiSpotsDataset):
            worker_seed = sampler_seed + worker_id
            sampler = WeightedDatasetSampler(self.dataset, num_samples=len(self.dataset), seed=worker_seed)
        else:
            raise ValueError('Dataset must be an instance of SpotsDataset or MultiSpotsDataset.')

        return sampler
    
    @staticmethod
    def _get_worker_info() -> Tuple[int, int]:

        info = torch.utils.data.get_worker_info()
        if info is None:
            worker_id, num_workers = 0, 1
        else:
            worker_id, num_workers = info.id, info.num_workers

        return worker_id, num_workers
    
    def __len__(self):

        _, num_workers = self._get_worker_info()
        num_samples = max(int(np.ceil(len(self.dataset) / num_workers)), self.batch_size)

        return num_samples

    def __iter__(self):

        # Get worker info.
        worker_id, num_workers = self._get_worker_info()
        num_samples = max(int(np.ceil(len(self.dataset) / num_workers)), self.batch_size)

        # Get random seeds.
        sq = np.random.SeedSequence((self.seed, self.epoch, worker_id))
        child_seeds = sq.generate_state(2, dtype=np.uint32)
        sampler_seed = int(child_seeds[0])
        augment_seed = int(child_seeds[1])

        # Make augment object.
        if self.augment_cls:
            augment = self.augment_cls(seed=augment_seed)
        else:
            augment = None

        if self.cached_indices is None:

            # Make sampler.
            sampler = self._make_sampler(sampler_seed, worker_id, num_workers)
            iterator = iter(sampler)
            
            # Yield samples.
            i = 0
            yielded = 0
            cached_indices = []
            while yielded < num_samples:

                if i >= len(sampler):
                    iterator = iter(sampler)
                    i = 0

                index = next(iterator)
                i += 1

                x, y = self.dataset[index]
                if augment:
                    x, y = augment.apply(x, y)
                if len(y) == 0:
                    continue
                if not self.shuffle:
                    cached_indices.append(index)
                yielded += 1

                yield x, y

            # Cache indices if not shuffling.
            if not self.shuffle:
                self.cached_indices = cached_indices

        else:
            
            for index in self.cached_indices:
                x, y = self.dataset[index]
                if augment:
                    x, y = augment.apply(x, y)
                yield x, y

        # Bump epoch.
        if self.shuffle:
            self.next_epoch()


def get_torch_dataloader(
        dataset: torch.utils.data.Dataset,
        image_size: Tuple[int, int],
        batch_size: int = 4,
        num_workers: int = 4,
        seed: int = 0,
        *args, **kwargs
) -> torch.utils.data.DataLoader:
    
    """Get a Torch dataloader from a dataset.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Torch dataset.
    image_size : Tuple[int, int]
        Desired image size.
    batch_size : int, optional
        Batch size. Default is 4.
    num_workers : int, optional
        Number of workers for data loading. Default is 4.
    seed : int, optional
        Random seed used for shuffling the dataset. Default is 0.

    Returns
    -------
    dataloader : torch.utils.data.DataLoader
        Torch dataloader.

    Raises
    ------
    ValueError
        If the dataset is not an instance of SpotsDataset or MultiSpotsDataset.
    """

    split = dataset.split
    if split == 'train':
        shuffle = True
        augment_cls = partial(RandomAugment, output_size=image_size, augment=True)
        drop_last = False
    else:
        shuffle = False
        augment_cls = partial(RandomAugment, output_size=image_size, augment=False)
        drop_last = False

    datastream = SpotsDataStream(dataset, batch_size, 1, seed, shuffle, augment_cls)
    dataloader = torch.utils.data.DataLoader(datastream, batch_size=batch_size, num_workers=num_workers,
                                             collate_fn=lambda x: list(map(list, zip(*x))), drop_last=drop_last,
                                             persistent_workers=num_workers > 0, *args, **kwargs)

    return dataloader


def get_torch_dataset(
        paths: Union[str, List[str], Dict[str, float], Path],
        adjustment: Optional[str] = 'standardize',
        load_train: bool = True,
        load_val: bool = True,
        load_test: bool = False,
) -> Dict:
    
    """Get a Torch dataset from a directory.

    Parameters
    ----------
    paths : Union[str, List[str], Dict[str, float], Path]
        Path to a dataset, path to a directory containing multiple datasets, a list of multiple dataset paths, or a
        dictionary of multiple dataset paths and their corresponding sampling weights. If a directory of datasets or a
        list is provided, all datasets in the directory or list will be loaded and concatenated with equal weights. If
        a dictionary is provided, the datasets will be loaded and concatenated with the specified weights.
    adjustment : Optional[str], optional
        Adjustment type applied to images. Supported types are 'normalize' and 'standardize'. Default is 'standardize'.
    load_train : bool, optional
        Whether to load the training set. Default is True.
    load_val : bool, optional
        Whether to load the validation set. Default is True.
    load_test : bool, optional
        Whether to load the test set. Default is False.

    Returns
    -------
    dataset : Dict
        Torch dataset.
    """

    # Intialize lists.
    subdatasets = []
    weights = None

    # Get Torch dataset.
    if isinstance(paths, (str, Path)):
        path = Path(paths)
        subdir_paths = sorted([p for p in path.glob('*') if p.is_dir()])
        if {'train', 'val', 'test'}.issubset((p.stem for p in subdir_paths)):
            dataset = {}
            for split, load_split in zip(('train', 'val', 'test'), (load_train, load_val, load_test)):
                    if load_split:
                        split_path = path / split
                        x_paths = sorted(list((split_path / 'x').glob('*.tif')))
                        y_paths = sorted(list((split_path / 'y').glob('*.csv')))
                        dataset[split] = SpotsDataset(x_paths, y_paths, adjustment, split)
            return dataset
        else:
            for subdir_path in subdir_paths:
                subdataset = get_torch_dataset(subdir_path, adjustment, load_train, load_val, load_test)
                subdatasets.append(subdataset)
    elif isinstance(paths, list):
        for path in paths:
            subdataset = get_torch_dataset(path, adjustment, load_train, load_val, load_test)
            subdatasets.append(subdataset)
    elif isinstance(paths, dict):
        weights = []
        for path, weight in paths.items():
            subdataset = get_torch_dataset(path, adjustment, load_train, load_val, load_test)
            subdatasets.append(subdataset)
            weights.append(weight)

    # Concatenate datasets.
    dataset = {}
    for split in subdatasets[0]:
        if weights:
            dataset[split] = MultiSpotsDataset([ds[split] for ds in subdatasets], weights)
        else:
            dataset[split] = torch.utils.data.ConcatDataset([ds[split] for ds in subdatasets])
            setattr(dataset[split], 'split', split)

    return dataset


def load_datasets(
        path: str,
        adjustment: Optional[str] = 'standardize',
        load_train: bool = True,
        load_val: bool = True,
        load_test: bool = True
) -> Dict:
    
    """Load datasets from a directory.

    Parameters
    ----------
    path : str
        Path to a dataset or directory of datasets.
    adjustment : Optional[str], optional
        Adjustment type applied to images. Supported types are 'normalize' and 'standardize'. Default is 'standardize'.
    load_train : bool, optional
        Whether to load the training set. Default is True.
    load_val : bool, optional
        Whether to load the validation set. Default is True.
    load_test : bool, optional
        Whether to load the test set. Default is True.

    Returns
    -------
    dataset : Dict
        Dataset dictionary.
    """

    # Get dataset paths.
    path = Path(path)
    subdir_paths = sorted(p for p in path.glob('*') if p.is_dir())

    # Load datasets.
    if {'train', 'val', 'test'}.issubset((p.stem for p in subdir_paths)):
        dataset = load_dataset(path, adjustment, load_train, load_val, load_test)
    else:
        datasets = []
        for subdir_path in subdir_paths:
            subdataset = load_datasets(subdir_path, adjustment, load_train, load_val, load_test)
            datasets.append(subdataset)
        dataset = {
            split: {'x': np.concatenate([subdataset[split]['x'] for subdataset in datasets]),
                    'y': np.concatenate([subdataset[split]['y'] for subdataset in datasets])}
            for split in datasets[0]
        }

    return dataset


def load_dataset(
        path: str,
        adjustment: Optional[str] = 'standardize',
        load_train: bool = True,
        load_val: bool = True,
        load_test: bool = True
) -> Dict:

    """Load a dataset from a directory.

    Parameters
    ----------
    path : str
        Path to a dataset.
    adjustment : Optional[str], optional
        Adjustment type applied to images. Supported types are 'normalize' and 'standardize'. Default is 'standardize'.
    load_train : bool, optional
        Whether to load the training set. Default is True.
    load_val : bool, optional
        Whether to load the validation set. Default is True.
    load_test : bool, optional
        Whether to load the test set. Default is True.

    Returns
    -------
    dataset : Dict
        Dataset dictionary.
    """

    # Intialize dataset dictionary.
    dataset = {}

    # Get dataset paths.
    path = Path(path)

    # Load dataset.
    for split, load_split in zip(('train', 'val', 'test'), (load_train, load_val, load_test)):

        if load_split:

            x_list = []
            y_list = []
            split_path = path / split

            for x_path in sorted((split_path / 'x').glob('*.tif')):
                y_path = split_path / 'y' / f'{x_path.stem}.csv'
                x = tifffile.imread(x_path)
                if y_path.stat().st_size == 0:
                    y = np.empty((0, 2))
                else:
                    y = np.loadtxt(y_path, delimiter=',').reshape(-1, 2)
                x_list.append(x)
                y_list.append(y)

            y_array = np.empty(len(y_list), dtype=object)
            y_array[:] = y_list
            dataset[split] = {'x': batch_adjust(x_list, adjustment), 'y': y_array}

    return dataset


def generate_dataset(
        path: str,
        images: List[np.ndarray],
        coords: List[np.ndarray],
        seed: int,
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
    seed : int
        Random seed used for splitting the dataset into training, validation, and test sets.
    tile_size : Tuple[int, int], optional
        Tile size used for splitting images. Default is (256, 256).
    min_spots : int, optional
        Minimum number of spots per tile. Default is 1.
    train_size : float, optional
        Fraction of dataset used for training. Default is 0.70.
    test_size : float, optional
        Fraction of dataset used for testing. Default is 0.15.
    """

    # Define paths.
    path = Path(path)
    x_train_path = path / 'train' / 'x'
    y_train_path = path / 'train' / 'y'
    x_val_path = path / 'val' / 'x'
    y_val_path = path / 'val' / 'y'
    x_test_path = path / 'test' / 'x'
    y_test_path = path / 'test' / 'y'
    x_train_path.mkdir(parents=True, exist_ok=True)
    y_train_path.mkdir(parents=True, exist_ok=True)
    x_val_path.mkdir(parents=True, exist_ok=True)
    y_val_path.mkdir(parents=True, exist_ok=True)
    x_test_path.mkdir(parents=True, exist_ok=True)
    y_test_path.mkdir(parents=True, exist_ok=True)

    # Remove duplicate coordinates.
    for i in range(len(coords)):
        coords[i] = remove_duplicate_coords(coords[i])

    tiled_images = []
    tiled_coords = []

    for image, c in zip(images, coords):

        # Create a DeepTile object and get tiles.
        dt = deeptile.load(image)
        tiles1 = dt.get_tiles(tile_size, (0, 0))
        tiles2 = tiles1.import_data(c, 'coords')
        nonempty_tiles = [(tile1.compute(), tile2)
                          for tile1, tile2 in zip(tiles1[tiles1.nonempty_indices], tiles2[tiles2.nonempty_indices])
                          if len(tile2) >= min_spots]
        tiles1, tiles2 = zip(*nonempty_tiles)
        tiled_images += tiles1
        tiled_coords += tiles2

    # Randomly shuffle the tiles.
    size = len(tiled_images)
    rng = np.random.default_rng(seed)
    perms = rng.permutation(size)

    # Split the dataset into training, validation, and test sets.
    split_indices = np.rint(np.cumsum((train_size, test_size)) * size).astype(int)
    train_padding = len(str(split_indices[0]))
    val_padding = len(str(len(tiled_images) - split_indices[1]))
    test_padding = len(str(split_indices[1] - split_indices[0]))
    for i, j in enumerate(perms[range(split_indices[0])]):
        tifffile.imwrite(x_train_path / f'{i:0{train_padding}d}.tif', tiled_images[j])
        np.savetxt(y_train_path / f'{i:0{train_padding}d}.csv', tiled_coords[j], delimiter=',')
    for i, j in enumerate(perms[range(split_indices[1], len(tiled_images))]):
        tifffile.imwrite(x_val_path / f'{i:0{val_padding}d}.tif', tiled_images[j])
        np.savetxt(y_val_path / f'{i:0{val_padding}d}.csv', tiled_coords[j], delimiter=',')
    for i, j in enumerate(perms[range(split_indices[0], split_indices[1])]):
        tifffile.imwrite(x_test_path / f'{i:0{test_padding}d}.tif', tiled_images[j])
        np.savetxt(y_test_path / f'{i:0{test_padding}d}.csv', tiled_coords[j], delimiter=',')
