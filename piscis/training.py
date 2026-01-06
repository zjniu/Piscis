import numpy as np
import sys
import torch

from tqdm import tqdm
from typing import Dict, List, Optional, Sequence, Tuple, Union

from piscis.convert import convert_jax_to_torch_state_dict
from piscis.data import get_torch_dataset, get_torch_dataloader
from piscis.downloads import download_pretrained_model
from piscis.losses import mean_masked_l2_loss, mean_smoothf1_loss
from piscis.models.spots import round_input_size, SpotsModel
from piscis.paths import CHECKPOINTS_DIR, MODELS_DIR
from piscis.transforms import batch_voronoi_transform


def loss_fn(
        labels_pred: torch.Tensor,
        deltas_pred: torch.Tensor,
        labels: torch.Tensor,
        deltas: torch.Tensor,
        p: torch.Tensor,
        l2_loss_weight: float,
        max_distance: float,
        kernel_size: Sequence[int],
        temperature: float,
        epsilon: float
) -> Tuple[torch.Tensor, Dict[str, float]]:
    
    """Computes the loss and metrics for a given batch.

    Parameters
    ----------
    labels_pred : torch.Tensor
        Predicted labels.
    deltas_pred : torch.Tensor
        Predicted displacement vectors.
    labels: torch.Tensor
        Ground truth labels.
    deltas : torch.Tensor
        Ground truth displacement vectors.
    p : torch.Tensor
        Number of ground truth spots in each image.
    l2_loss_weight : float
        Weight for the masked L2 loss term in the overall loss function.
    max_distance : float
        Maximum distance for matching predicted and ground truth displacement vectors.
    kernel_size : Sequence[int], optional
        Kernel size of sum or max pooling operations. Default is (3, 3).
    temperature : float
        Temperature parameter.
    epsilon : float
        Small constant for numerical stability.

    Returns
    -------
    loss : torch.Tensor
        Overall loss value.
    metrics : Dict[str, float]
        Dictionary containing the values of individual loss terms and the overall loss.
    """

    # Compute SmoothF1 loss and L2 loss.
    smoothf1 = mean_smoothf1_loss(labels_pred, deltas_pred, deltas, p,
                                  max_distance, kernel_size, temperature, epsilon)
    l2 = mean_masked_l2_loss(deltas_pred, deltas, labels, epsilon)

    # Combine losses.
    loss = smoothf1 + l2_loss_weight * l2

    # Create metrics dictionary.
    metrics = {
        'loss': loss.detach().item(),
        'smoothf1': smoothf1.detach().item(),
        'l2': l2.detach().item()
    }

    return loss, metrics


def train_epoch(
        model: SpotsModel,
        dataloader: tqdm,
        optimizer: torch.optim.Optimizer,
        l2_loss_weight: float,
        dilation_iterations: int,
        max_distance: float,
        temperature: float,
        epsilon: float,
        device: Optional[str]
) -> Dict[str, float]:

    """Train the model for a single epoch.

    Parameters
    ----------
    model : SpotsModel
        Model to be trained.
    dataloader : tqdm
        DataLoader for the training data.
    optimizer : torch.optim.Optimizer
        Optimizer for updating model parameters.
    l2_loss_weight : float
        Weight for the masked L2 loss term in the overall loss function.
    dilation_iterations : int
        Number of iterations to dilate ground truth labels
    max_distance : float
        Maximum distance for matching predicted and ground truth displacement vectors.
    temperature : float
        Temperature parameter for softmax.
    epsilon : float
        Small constant for numerical stability.
    device : Optional[str]
        Device for training.

    Returns
    -------
    train_metrics : Dict[str, float]
        Dictionary containing average training metrics for the epoch.
    """

    # Initialize dictionary for running metrics.
    running_metrics = {}

    # Set model to training mode.
    model.train()

    for i, batch in enumerate(dataloader):

        # Apply Voronoi transform to the batch.
        x, y = batch
        x = torch.tensor(np.stack(x), dtype=torch.float).to(device)
        labels, deltas = batch_voronoi_transform(y, x.shape[-2:], dilation_iterations, device)
        p = torch.tensor([len(coords) for coords in y], dtype=torch.float, device=device)

        # Train the model on the batch.
        optimizer.zero_grad()
        labels_pred, deltas_pred = model(x)
        loss, metrics = loss_fn(labels_pred, deltas_pred, labels, deltas, p, l2_loss_weight,
                                max_distance, model.kernel_size, temperature, epsilon)
        loss.backward()
        optimizer.step()
        running_metrics = {k: running_metrics.get(k, 0.0) + metrics[k] for k in metrics}

        # Update progress bar.
        mean_metrics = {k: v / (i + 1) for k, v in running_metrics.items()}
        dataloader.set_postfix({k: f'{mean_metrics[k]:.4f}' for k in ('loss', 'smoothf1', 'l2')})

    train_metrics = mean_metrics

    return train_metrics


def val_epoch(
        model: SpotsModel,
        dataloader: torch.utils.data.DataLoader,
        l2_loss_weight: float,
        dilation_iterations: int,
        max_distance: float,
        temperature: float,
        epsilon: float,
        device: Optional[str]
) -> Dict[str, float]:

    """Validate the model for a single epoch.

    Parameters
    ----------
    model : SpotsModel
        Model to be validated.
    dataloader : torch.utils.data.DataLoader
        DataLoader for the validation data.
    l2_loss_weight : float
        Weight for the masked L2 loss term in the overall loss function.
    dilation_iterations : int
        Number of iterations to dilate ground truth labels
    max_distance : float
        Maximum distance for matching predicted and ground truth displacement vectors.
    temperature : float
        Temperature parameter for softmax.
    epsilon : float
        Small constant for numerical stability.
    device : Optional[str]
        Device for training.

    Returns
    -------
    val_metrics : Dict[str, float]
        Dictionary containing average validation metrics for the epoch.
    """

    # Initialize dictionary for running metrics.
    running_metrics = {}

    # Set model to eval mode.
    model.eval()

    with torch.no_grad():

        for batch in dataloader:

            # Apply Voronoi transform to the batch.
            x, y = batch
            x = torch.tensor(np.stack(x), dtype=torch.float).to(device)
            labels, deltas = batch_voronoi_transform(y, x.shape[-2:], dilation_iterations, device)
            p = torch.tensor([len(coords) for coords in y], dtype=torch.float, device=device)

            # Validate the model on the batch.
            labels_pred, deltas_pred = model(x)
            _, metrics = loss_fn(labels_pred, deltas_pred, labels, deltas, p, l2_loss_weight,
                                 max_distance, model.kernel_size, temperature, epsilon)
            running_metrics = {k: running_metrics.get(k, 0.0) + metrics[k] for k in metrics}

    val_metrics = {f'val_{k}': v / len(dataloader) for k, v in running_metrics.items()}

    return val_metrics


def train_model(
        model_name: str,
        dataset_path: Union[str, List[str], Dict[str, float]],
        initial_model_name: Optional[str] = None,
        adjustment: Optional[str] = 'standardize',
        input_size: Tuple[int, int] = (256, 256),
        random_seed: int = 0,
        batch_size: int = 4,
        num_workers: int = 0,
        learning_rate: float = 0.1,
        weight_decay: float = 1e-5,
        epochs: int = 500,
        warmup_fraction: float = 0.04,
        decay_fraction: float = 0.4,
        decay_transitions: int = 10,
        decay_factor: float = 0.5,
        l2_loss_weight: float = 0.1,
        dilation_iterations: int = 1,
        max_distance: float = 3.0,
        temperature: float = 0.05,
        epsilon: float = 1e-7,
        checkpoint_every: int = 10,
        device: Optional[str] = 'cuda'
) -> None:

    """Train a SpotsModel.

    Parameters
    ----------
    model_name : str
        Model name.
    dataset_path : Union[str, List[str], Dict[str, float]]
        Path to a dataset, path to a directory containing multiple datasets, a list of multiple dataset paths, or a
        dictionary of multiple dataset paths and their corresponding sampling weights. If a directory of datasets or a
        list is provided, all datasets in the directory or list will be loaded and concatenated with equal weights. If
        a dictionary is provided, the datasets will be loaded and concatenated with the specified weights.
    initial_model_name : Optional[str], optional
        Name of an existing model to initialize the weights. Default is None.
    adjustment : Optional[str], optional
        Adjustment type applied to images. Supported types are 'normalize' and 'standardize'. Default is 'standardize'.
    input_size : Tuple[int, int], optional
        Input size used for training. Default is (256, 256).
    random_seed : int, optional
        Random seed used for initialization and training. Default is 0.
    batch_size : int, optional
        Batch size for training. Default is 4.
    num_workers : int, optional
        Number of workers for data loading. Default is 0.
    learning_rate : float, optional
        Learning rate for the optimizer. Default is 0.1.
    weight_decay : float, optional
        Strength of the weight decay regularization. Default is 1e-5.
    epochs : int, optional
        Number of epochs to train the model for. Default is 500.
    warmup_fraction : float, optional
        Fraction of epochs for learning rate warmup. Default is 0.04.
    decay_fraction : float, optional
        Fraction of epochs for learning rate decay. Default is 0.4.
    decay_transitions : int, optional
        Number of times to decay the learning rate. Default is 10.
    decay_factor : float, optional
        Multiplicative factor of each learning rate decay transition. Default is 0.5.
    l2_loss_weight : float, optional
        Weight for the masked L2 loss term in the overall loss function. Default is 0.1.
    dilation_iterations : int, optional
        Number of iterations to dilate ground truth labels to minimize class imbalance and misclassifications due to
        minor offsets. Default is 1.
    max_distance : float, optional
        Maximum distance for matching predicted and ground truth displacement vectors. Default is 3.0.
    temperature : float, optional
        Temperature parameter for softmax. Default is 0.05.
    epsilon : float, optional
        Small constant for numerical stability. Default is 1e-7.
    checkpoint_every : int, optional
        Number of epochs between saving model checkpoints. Default is 10.
    device : Optional[str], optional
        Device for training. Default is 'cuda'.

    Raises
    ------
    ValueError
        If warmup_fraction + decay_fraction is greater than 1.
    """

    if warmup_fraction + decay_fraction > 1:
        raise ValueError("warmup_fraction + decay_fraction cannot be greater 1.")
    
    # Split random seed.
    sq = np.random.SeedSequence(random_seed)
    child_seeds = sq.generate_state(4, dtype=np.uint32)

    # Round the input image size.
    input_size = round_input_size(input_size)

    # Get dataloaders.
    dataset = get_torch_dataset(dataset_path, adjustment, load_train=True, load_val=True, load_test=False)
    train_dataloader = get_torch_dataloader(dataset['train'], image_size=input_size, batch_size=batch_size,
                                            num_workers=num_workers, seed=int(child_seeds[0]))
    val_dataloader = get_torch_dataloader(dataset['val'], image_size=input_size, batch_size=batch_size,
                                          num_workers=num_workers, seed=int(child_seeds[1]))
    channels = next(iter(train_dataloader))[0][0].shape[0]

    # Create the model.
    torch.manual_seed(child_seeds[2])
    kernel_size = (2 * dilation_iterations + 1, ) * 2
    model = SpotsModel(in_channels=channels, pooling='max', kernel_size=kernel_size).to(device)

    # Create the optimizer.
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9,
                                weight_decay=weight_decay, nesterov=True)

    # Create the learning rate schedule.
    warmup_epochs = round(warmup_fraction * epochs)
    decay_epochs = round(decay_fraction * epochs)
    decay_transition_epochs = int(np.ceil(decay_epochs / decay_transitions))
    decay_epochs = decay_epochs + decay_transition_epochs
    decay_milestone = epochs - decay_epochs
    constant_epochs = decay_milestone - warmup_epochs
    schedulers = []
    milestones = []
    if warmup_epochs > 0:
        linear_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-7, end_factor=1.0, total_iters=warmup_epochs
        )
        schedulers.append(linear_scheduler)
        milestones.append(warmup_epochs)
    if constant_epochs > 0:
        constant_scheduler = torch.optim.lr_scheduler.ConstantLR(
            optimizer,
            factor=1.0, total_iters=constant_epochs
        )
        schedulers.append(constant_scheduler)
        milestones.append(decay_milestone)
    if decay_epochs > 0:
        decay_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=decay_transition_epochs, gamma=decay_factor
        )
        schedulers.append(decay_scheduler)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=schedulers,milestones=milestones)

    # Define directories for storing checkpoints and the model.
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    last_checkpoint_path = CHECKPOINTS_DIR / f'{model_name}_last.pt'
    best_checkpoint_path = CHECKPOINTS_DIR / f'{model_name}_best.pt'

    model_path = MODELS_DIR / f'{model_name}.pt'

    if last_checkpoint_path.is_file():

        # Load the last checkpoint if available.
        print(f'Loading the last checkpoint from {last_checkpoint_path}...')
        checkpoint = torch.load(last_checkpoint_path, map_location='cpu')
        last_epoch = checkpoint['epoch']
        train_dataloader.dataset.set_epoch(last_epoch + 1)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        metrics_log = checkpoint['metrics_log']
        best_val_loss = min(metrics.get('val_loss', float('inf')) for metrics in metrics_log)

    else:

        last_epoch = 0

        # Initialize model weights.
        if initial_model_name:
            initial_model_path = MODELS_DIR / f'{initial_model_name}.pt'
            if not initial_model_path.is_file():
                if (MODELS_DIR / initial_model_name).is_file():
                    state_dict = convert_jax_to_torch_state_dict(initial_model_name)
                    torch.save(state_dict, initial_model_path)
                else:
                    download_pretrained_model(initial_model_name)
            print(f'Initializing model weights from {initial_model_path}...')
            with open(initial_model_path, 'rb') as f_model:
                state_dict = torch.load(f_model, map_location='cpu')
                state_dict.pop('metadata')
                first_layer = state_dict['fpn.encoder.stem.conv.0.weight']
                if (first_layer.shape[1] == 1) and (channels > 1):
                    state_dict['fpn.encoder.stem.conv.0.weight'] = torch.cat([first_layer] * channels, dim=1)
                if first_layer.shape[1] > channels:
                    state_dict['fpn.encoder.stem.conv.0.weight'] = first_layer[:, :channels, :, :]
                model.load_state_dict(state_dict)

        # Create list for storing metrics.
        metrics_log = []
        best_val_loss = float('inf')

    # Determine epoch string length.
    epoch_str_len = len(str(epochs))

    # Train the model.
    for epoch in range(last_epoch + 1, epochs + 1):

        # Create progress bar.
        pbar_desc = f'Epoch {epoch:0{epoch_str_len}}'
        pbar = tqdm(
            train_dataloader,
            desc=pbar_desc,
            leave=False,
            file=sys.stdout
        )

        # Train for a single epoch.
        train_metrics = train_epoch(model, pbar, optimizer, l2_loss_weight, dilation_iterations, max_distance,
                                    temperature, epsilon, device)
        metrics = {'epoch': epoch, 'lr': optimizer.param_groups[0]['lr']} | train_metrics
        train_summary = f'{pbar_desc}: ' + \
                        ', '.join([f'{k}={train_metrics[k]:.4f}' for k in ('loss', 'smoothf1', 'l2')])

        if len(val_dataloader) > 0:

            print(train_summary, end='', flush=True)

            # Validate for a single epoch.
            val_metrics = val_epoch(model, val_dataloader, l2_loss_weight, dilation_iterations, max_distance,
                                    temperature, epsilon, device)
            val_loss = val_metrics['val_loss']
            metrics = metrics | val_metrics
            val_summary = ' | ' + ', '.join([f'{k}={val_metrics[k]:.4f}' for k in ('val_loss', 'val_smoothf1', 'val_l2')])
            print(f'\r{train_summary + val_summary}')

        else:

            print(train_summary)

            val_loss = float('inf')

        # Update metrics log.
        metrics_log.append(metrics)

        # Step the learning rate scheduler.
        scheduler.step()

        # Save the best checkpoint if necessary.
        best = val_loss < best_val_loss
        if best and (epoch >= decay_milestone):
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'metrics_log': metrics_log,
                'best_val_loss': best_val_loss
            }
            torch.save(checkpoint, best_checkpoint_path)
        
        # Save a checkpoint if necessary.
        if best or (epoch % checkpoint_every == 0) or (epoch == epochs):
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'metrics_log': metrics_log,
                'best_val_loss': best_val_loss
            }
            torch.save(checkpoint, last_checkpoint_path)

    # Save the best model.
    if best_checkpoint_path.is_file():
        state_dict = torch.load(best_checkpoint_path, map_location='cpu')['model_state_dict']
    else:
        state_dict = torch.load(last_checkpoint_path, map_location='cpu')['model_state_dict']
    state_dict['metadata'] = {
        'adjustment': adjustment,
        'input_size': input_size,
        'dilation_iterations': dilation_iterations,
        'channels': channels
    }
    torch.save(state_dict, model_path)
