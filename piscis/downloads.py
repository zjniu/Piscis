from huggingface_hub import HfFileSystem
from typing import Sequence

from piscis.paths import HF_MODELS_DIR, MODELS_DIR


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
    hf_model_path = f'{HF_MODELS_DIR}{model_name}'
    if fs.exists(hf_model_path):
        fs.download(hf_model_path, str(MODELS_DIR / model_name))
    else:
        raise ValueError(f'Model {model_name} is not found.')


def list_pretrained_models() -> Sequence[str]:

    """List pretrained models from Hugging Face.

    Returns
    -------
    pretrained_models : Sequence[str]
        List of pretrained model names.
    """

    fs = HfFileSystem()
    pretrained_models = [path.removeprefix(HF_MODELS_DIR) for path in fs.ls(HF_MODELS_DIR, detail=False)]

    return pretrained_models
