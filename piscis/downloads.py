from huggingface_hub import HfFileSystem

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

    # Get a list of pretrained models from Hugging Face.
    fs = HfFileSystem()
    pretrained_models = fs.ls(HF_MODELS_DIR, detail=False)

    # Download the model if available.
    hf_model_path = f'{HF_MODELS_DIR}/{model_name}'
    if hf_model_path in pretrained_models:
        fs.download(hf_model_path, str(MODELS_DIR / model_name))
    else:
        raise ValueError(f'Model {model_name} is not found.')
