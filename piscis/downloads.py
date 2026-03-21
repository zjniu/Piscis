from huggingface_hub import HfFileSystem
from pathlib import Path
from typing import Sequence

from piscis.paths import HF_DATASETS_DIR, HF_MODELS_DIR, MODELS_DIR


def download_pretrained_model(
        model_name: str
) -> None:

    """Download a pretrained model from Hugging Face.

    Parameters
    ----------
    model_name : str
        Model name.

    Raises
    ------
    ValueError
        If `model_name` is not found.
    """

    # Create the models directory if necessary.
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Download the model if available.
    fs = HfFileSystem()
    hf_model_path = f'{HF_MODELS_DIR}{model_name}.pt'
    if fs.exists(hf_model_path):
        fs.download(hf_model_path, str(MODELS_DIR / f'{model_name}.pt'))
    else:
        raise ValueError(f"Model {model_name} is not found.")


def list_pretrained_models() -> Sequence[str]:

    """List pretrained models from Hugging Face.

    Returns
    -------
    pretrained_models : Sequence[str]
        List of pretrained model names.
    """

    fs = HfFileSystem()
    pretrained_models = set(path.removeprefix(HF_MODELS_DIR).rsplit('.pt', 1)[0]
                            for path in fs.ls(HF_MODELS_DIR, detail=False) if path.endswith('.pt'))

    return pretrained_models


def download_dataset(
        dataset_name: str,
        path: str,
        minimal_download: bool = True
) -> None:

    """Download a dataset from Hugging Face.

    Parameters
    ----------
    dataset_name : str
        Dataset name.
    path : str
        Path to save the dataset.

    Raises
    ------
    ValueError
        If `dataset_name` is not found.
    """

    fs = HfFileSystem()
    hf_dataset_path = f'{HF_DATASETS_DIR}{dataset_name}'
    if fs.exists(hf_dataset_path) and ('/' not in dataset_name):
        fs.download(hf_dataset_path, str(path), recursive=True)
    else:
        raise ValueError(f"Dataset {dataset_name} is not found.")


def list_datasets() -> Sequence[str]:

    """List datasets from Hugging Face.

    Returns
    -------
    datasets : Sequence[str]
        List of dataset names.
    """

    fs = HfFileSystem()
    datasets = [path.removeprefix(HF_DATASETS_DIR) for path in fs.ls(HF_DATASETS_DIR, detail=False) if fs.isdir(path)]

    return datasets
