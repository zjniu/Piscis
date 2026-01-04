import numpy as np
import tifffile
import torch

from flax import serialization
from pathlib import Path

from piscis.paths import MODELS_DIR

def convert_jax_to_torch_state_dict(jax_model_name, state_dict=None, verbose=False):

    """Convert Piscis JAX model weights to PyTorch state dict.

    Parameters
    ----------
    jax_model_name : str
        JAX model name.
    state_dict : dict
        Template state dict from PyTorch model. Default is None.
    verbose : bool, optional
        Whether to print conversion progress. Default is False.

    Raises
    ------
    ValueError
        If there is a shape mismatch between JAX and PyTorch weights.
    """

    # Get state dict.
    if state_dict is None:
        check_shapes = False
        state_dict = {}
    else:
        check_shapes = True
    metadata = {}
    state_dict['metadata'] = metadata

    # Load JAX model.
    model_path = MODELS_DIR / jax_model_name
    with open(model_path, 'rb') as f_model:
        jax_dict = serialization.from_bytes(target=None, encoded_bytes=f_model.read())
        jax_variables = jax_dict['variables']
        metadata['adjustment'] = jax_dict.get('adjustment', None)
        input_size = jax_dict.get('input_size', None)
        if input_size is not None:
            input_size = (input_size['0'], input_size['1'])
        metadata['input_size'] = input_size
        metadata['dilation_iterations'] = jax_dict.get('dilation_iterations', None)
        metadata['channels'] = jax_dict.get('channels', 1)

    # Map EfficientNetV2 encoder weights.
    encoder_mapping = {}

    # Stem.
    mapping = {
        '0.weight': ['stem_conv', 'kernel'],
        '1.weight': ['stem_bn', 'scale'],
        '1.bias': ['stem_bn', 'bias'],
        '1.running_mean': ['stem_bn', 'mean'],
        '1.running_var': ['stem_bn', 'var']
    }
    mapping = {f'stem.conv.{k}': v for k, v in mapping.items()}
    encoder_mapping = encoder_mapping | mapping

    # FusedMBConv 0-3.
    for i in range(4):
        mapping = {
            '0.weight': ['project_conv', 'kernel'],
            '1.weight': ['project_bn', 'scale'],
            '1.bias': ['project_bn', 'bias'],
            '1.running_mean': ['project_bn', 'mean'],
            '1.running_var': ['project_bn', 'var']
        }
        mapping = {f'blocks.0.{i}.block.0.conv.{k}': [f'FusedMBConv_{i}'] + v for k, v in mapping.items()}
        encoder_mapping = encoder_mapping | mapping

    # FusedMBConv 4-7.
    for i in range(4):
        mapping = {
            '0.conv.0.weight': ['expand_conv', 'kernel'],
            '0.conv.1.weight': ['expand_bn', 'scale'],
            '0.conv.1.bias': ['expand_bn', 'bias'],
            '0.conv.1.running_mean': ['expand_bn', 'mean'],
            '0.conv.1.running_var': ['expand_bn', 'var'],
            '1.conv.0.weight': ['project_conv', 'kernel'],
            '1.conv.1.weight': ['project_bn', 'scale'],
            '1.conv.1.bias': ['project_bn', 'bias'],
            '1.conv.1.running_mean': ['project_bn', 'mean'],
            '1.conv.1.running_var': ['project_bn', 'var']
        }
        mapping = {f'blocks.1.{i + 1}.block.{k}': [f'FusedMBConv_{i + 4}'] + v for k, v in mapping.items()}
        encoder_mapping = encoder_mapping | mapping

    # MBConv 0-7.
    for i in range(2):
        for j in range(4):
            l = i * 4 + j
            mapping = {
                '0.conv.0.weight': ['expand_conv', 'kernel'],
                '0.conv.1.weight': ['expand_bn', 'scale'],
                '0.conv.1.bias': ['expand_bn', 'bias'],
                '0.conv.1.running_mean': ['expand_bn', 'mean'],
                '0.conv.1.running_var': ['expand_bn', 'var'],
                '1.conv.0.weight': ['dw_conv', 'kernel'],
                '1.conv.1.weight': ['dw_bn', 'scale'],
                '1.conv.1.bias': ['dw_bn', 'bias'],
                '1.conv.1.running_mean': ['dw_bn', 'mean'],
                '1.conv.1.running_var': ['dw_bn', 'var'],
                '2.reduce.weight': ['se_reduce', 'kernel'],
                '2.reduce.bias': ['se_reduce', 'bias'],
                '2.expand.weight': ['se_expand', 'kernel'],
                '2.expand.bias': ['se_expand', 'bias'],
                '3.conv.0.weight': ['project_conv', 'kernel'],
                '3.conv.1.weight': ['project_bn', 'scale'],
                '3.conv.1.bias': ['project_bn', 'bias'],
                '3.conv.1.running_mean': ['project_bn', 'mean'],
                '3.conv.1.running_var': ['project_bn', 'var']
            }
            mapping = {f'blocks.{i + 2}.{j + 1}.block.{k}': [f'MBConv_{l}'] + v for k, v in mapping.items()}
            encoder_mapping = encoder_mapping | mapping

    encoder_mapping = {f'fpn.encoder.{k}': ['FPN_0', 'EfficientNetV2XS'] + v for k, v in encoder_mapping.items()}

    # Map decoder weights.
    decoder_mapping = {}

    # Define up mapping.
    up_mapping = {
        'proj.conv.0.weight': ['Conv_0', 'BatchNorm_0', 'scale'],
        'proj.conv.0.bias': ['Conv_0', 'BatchNorm_0', 'bias'],
        'proj.conv.0.running_mean': ['Conv_0', 'BatchNorm_0', 'mean'],
        'proj.conv.0.running_var': ['Conv_0', 'BatchNorm_0', 'var'],
        'proj.conv.1.weight': ['Conv_0', 'Conv_0', 'kernel'],
        'proj.conv.1.bias': ['Conv_0', 'Conv_0', 'bias'],
        'conv.conv.0.weight': ['Conv_1', 'BatchNorm_0', 'scale'],
        'conv.conv.0.bias': ['Conv_1', 'BatchNorm_0', 'bias'],
        'conv.conv.0.running_mean': ['Conv_1', 'BatchNorm_0', 'mean'],
        'conv.conv.0.running_var': ['Conv_1', 'BatchNorm_0', 'var'],
        'conv.conv.2.weight': ['Conv_1', 'Conv_0', 'kernel'],
        'conv.conv.2.bias': ['Conv_1', 'Conv_0', 'bias']
    }

    # Define BatchConvStyle mapping.
    bcs_mapping = {
        'conv.conv.0.weight': ['Conv_0', 'BatchNorm_0', 'scale'],
        'conv.conv.0.bias': ['Conv_0', 'BatchNorm_0', 'bias'],
        'conv.conv.0.running_mean': ['Conv_0', 'BatchNorm_0', 'mean'],
        'conv.conv.0.running_var': ['Conv_0', 'BatchNorm_0', 'var'],
        'conv.conv.2.weight': ['Conv_0', 'Conv_0', 'kernel'],
        'conv.conv.2.bias': ['Conv_0', 'Conv_0', 'bias'],
        'dense.weight': ['Dense_0', 'kernel'],
        'dense.bias': ['Dense_0', 'bias']
    }

    # Up blocks.
    for i in range(4):
        mapping = {f'up_blocks.{i}.{k}': [f'UpConv_{i}'] + v for k, v in up_mapping.items()}
        decoder_mapping = decoder_mapping | mapping
        for j, l in ((0, 0), (1, 2), (2, 1)):
            mapping = {f'up_blocks.{i}.convs_{j}.{k}': [f'UpConv_{i}', f'BatchConvStyle_{l}'] + v for k, v in bcs_mapping.items()}
            decoder_mapping = decoder_mapping | mapping

    # Resize up blocks.
    for i, m, n in [(0, 0, 4), (0, 1, 5), (0, 2, 6), (1, 0, 7), (1, 1, 8), (2, 0, 9)]:
        mapping = {f'resize_up_blocks.{i}.{m}.{k}': [f'UpConv_{n}'] + v for k, v in up_mapping.items()}
        decoder_mapping = decoder_mapping | mapping
        for j, l in ((0, 0), (1, 2), (2, 1)):
            mapping = {f'resize_up_blocks.{i}.{m}.convs_{j}.{k}': [f'UpConv_{n}', f'BatchConvStyle_{l}'] + v for k, v in bcs_mapping.items()}
            decoder_mapping = decoder_mapping | mapping

    decoder_mapping = {f'fpn.decoder.{k}': ['FPN_0', 'Decoder_0'] + v for k, v in decoder_mapping.items()}

    # Map output conv weights.
    output_mapping = {
        '2.weight': ['Conv_0', 'kernel'],
        '2.bias': ['Conv_0', 'bias'],
        '0.weight': ['BatchNorm_0', 'scale'],
        '0.bias': ['BatchNorm_0', 'bias'],
        '0.running_mean': ['BatchNorm_0', 'mean'],
        '0.running_var': ['BatchNorm_0', 'var']
    }
    output_mapping = {f'fpn.output.conv.{k}': ['FPN_0', 'Conv_0'] + v for k, v in output_mapping.items()}

    # Combine all mappings.
    fpn_mapping = encoder_mapping | decoder_mapping | output_mapping

    # Map weights from JAX to PyTorch.
    for k, v in fpn_mapping.items():

        bs = jax_variables['batch_stats']

        if v[-1] in ('mean', 'var'):
            bs = jax_variables['batch_stats']
            for key in v:
                bs = bs[key]
            w_jax = bs
        else:
            p = jax_variables['params']
            for key in v:
                p = p[key]
            w_jax = p
        w_torch = torch.from_numpy(np.array(w_jax))

        if v[-1] == 'kernel':
            if 'dense' in k:
                w_torch = w_torch.permute(1, 0)
            else:
                w_torch = w_torch.permute(3, 2, 0, 1)

        if k.startswith('fpn.output.conv.2'):
            w_torch = torch.concatenate([w_torch[2:], w_torch[:2]], dim=0)
        
        if check_shapes and (state_dict[k].shape != w_torch.shape):
            raise ValueError(f"Shape mismatch for {k}: {state_dict[k].shape} vs {w_torch.shape}")
            
        state_dict[k] = w_torch
        if verbose:
            print(f"{k} loaded from {'/'.join(v)}.")

    print(f"Converted JAX model {jax_model_name} to PyTorch.")

    return state_dict


def convert_dataset(
        dataset_path: str,
        new_dataset_path: str
) -> None:

    """Convert a Piscis dataset saved as a .npz file to directories of .tif and .csv files.
    
    Parameters
    ----------
    dataset_path : str
        Path to the .npz dataset file.
    new_dataset_path : str
        Path to the directory for the converted dataset.
    """

    # Load dataset.
    npz = np.load(dataset_path, allow_pickle=True)
    x_train = npz['x_train']
    y_train = npz['y_train']
    x_val = npz['x_valid']
    y_val = npz['y_valid']
    x_test = npz['x_test']
    y_test = npz['y_test']

    # Define paths.
    new_dataset_path = Path(new_dataset_path)
    x_train_path = new_dataset_path / 'train' / 'x'
    y_train_path = new_dataset_path / 'train' / 'y'
    x_val_path = new_dataset_path / 'val' / 'x'
    y_val_path = new_dataset_path / 'val' / 'y'
    x_test_path = new_dataset_path / 'test' / 'x'
    y_test_path = new_dataset_path / 'test' / 'y'
    x_train_path.mkdir(parents=True, exist_ok=True)
    y_train_path.mkdir(parents=True, exist_ok=True)
    x_val_path.mkdir(parents=True, exist_ok=True)
    y_val_path.mkdir(parents=True, exist_ok=True)
    x_test_path.mkdir(parents=True, exist_ok=True)
    y_test_path.mkdir(parents=True, exist_ok=True)

    # Save converted dataset.
    train_padding = len(str(len(x_train)))
    val_padding = len(str(len(x_val)))
    test_padding = len(str(len(x_test)))
    for i, j in enumerate(range(len(x_train))):
        tifffile.imwrite(x_train_path / f'{i:0{train_padding}d}.tif', x_train[j])
        np.savetxt(y_train_path / f'{i:0{train_padding}d}.csv', y_train[j], delimiter=',')
    for i, j in enumerate(range(len(x_val))):
        tifffile.imwrite(x_val_path / f'{i:0{val_padding}d}.tif', x_val[j])
        np.savetxt(y_val_path / f'{i:0{val_padding}d}.csv', y_val[j], delimiter=',')
    for i, j in enumerate(range(len(x_test))):
        tifffile.imwrite(x_test_path / f'{i:0{test_padding}d}.tif', x_test[j])
        np.savetxt(y_test_path / f'{i:0{test_padding}d}.csv', y_test[j], delimiter=',')
