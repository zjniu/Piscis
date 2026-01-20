import argparse
import imageio.v3 as iio
import numpy as np

from pathlib import Path
from tqdm import tqdm

from piscis import Piscis
from piscis.training import train_model


def main():

    """Main function for the Piscis command-line interface."""

    parser = argparse.ArgumentParser(description="Piscis Algorithm")
    subparsers = parser.add_subparsers(dest='command')
    formatter_class = argparse.ArgumentDefaultsHelpFormatter

    # Predict subparser.
    predict_parser = subparsers.add_parser('predict', formatter_class=formatter_class,
                                           help="Predict spots using a trained SpotsModel.")
    predict_parser.add_argument('input_path', type=str,
                                help="Path to image or folder of images.")
    predict_parser.add_argument('output_path', type=str,
                                help="Path to save predicted spots as CSV.")
    predict_parser.add_argument('--model-name', type=str, default='20251212',
                                help="Model name.")
    predict_parser.add_argument('--batch-size', type=int, default=1,
                                help="Batch size for the CNN.")
    predict_parser.add_argument('--stack', action='store_true',
                                help="Whether the input is a stack of images.")
    predict_parser.add_argument('--scale', type=float, default=1.0,
                                help="Scale factor for rescaling the input.")
    predict_parser.add_argument('--threshold', type=float, default=0.5,
                                help="Spot detection threshold.")
    predict_parser.add_argument('--min-distance', type=int, default=1,
                                help="Minimum distance between spots.")

    # Train subparser.
    train_parser = subparsers.add_parser('train', formatter_class=formatter_class,
                                         help="Train a SpotsModel")
    train_parser.add_argument('model_name', type=str,
                              help="Model name.")
    train_parser.add_argument('dataset_path', type=str,
                              help="Path to a dataset or path to a directory containing multiple datasets.")
    train_parser.add_argument('--initial-model-name', type=str, default=None,
                              help="Name of an existing model to initialize the weights.")
    train_parser.add_argument('--adjustment', type=str, default='standardize',
                              help="Adjustment type applied to images. Supported types are 'normalize' and "
                                   "'standardize'.")
    train_parser.add_argument('--input-size', type=int, nargs=2, default=(256, 256),
                              help="Input size used for training.")
    train_parser.add_argument('--random-seed', type=int, default=0,
                              help="Random seed used for initialization and training.")
    train_parser.add_argument('--batch-size', type=int, default=4,
                              help="Batch size for training.")
    train_parser.add_argument('--num-workers', type=int, default=0,
                              help="Number of workers for data loading.")
    train_parser.add_argument('--learning-rate', type=float, default=0.1,
                              help="Learning rate for the optimizer.")
    train_parser.add_argument('--weight-decay', type=float, default=1e-5,
                              help="Strength of the weight decay regularization.")
    train_parser.add_argument('--epochs', type=int, default=500,
                              help="Number of epochs to train the model for.")
    train_parser.add_argument('--warmup-fraction', type=float, default=0.04,
                              help="Fraction of epochs for learning rate warmup.")
    train_parser.add_argument('--decay-fraction', type=float, default=0.4,
                              help="Fraction of epochs for learning rate decay.")
    train_parser.add_argument('--l2-loss-weight', type=float, default=0.1,
                              help="Weight for the masked L2 loss term in the overall loss function.")
    train_parser.add_argument('--decay-transitions', type=int, default=10,
                              help="Number of times to decay the learning rate.")
    train_parser.add_argument('--decay-factor', type=float, default=0.5,
                              help="Multiplicative factor of each learning rate decay transition.")
    train_parser.add_argument('--dilation-iterations', type=int, default=1,
                              help="Number of iterations to dilate ground truth labels to minimize class imbalance "
                                   "and misclassifications due to minor offsets.")
    train_parser.add_argument('--max-distance', type=float, default=3.0,
                              help="Maximum distance for matching predicted and ground truth displacement vectors.")
    train_parser.add_argument('--temperature', type=float, default=0.05,
                              help="Temperature parameter for softmax.")
    train_parser.add_argument('--epsilon', type=float, default=1e-7,
                              help="Small constant for numerical stability.")
    train_parser.add_argument('--checkpoint_every', type=int, default=10,
                              help="Number of epochs between saving model checkpoints.")
    train_parser.add_argument('--device', type=str, default='cuda',
                              help="Device for training.")

    args = parser.parse_args()

    if args.command == 'train':

        # Train the model.
        train_model(
            model_name=args.model_name,
            dataset_path=args.dataset_path,
            initial_model_name=args.initial_model_name,
            adjustment=args.adjustment,
            input_size=args.input_size,
            random_seed=args.random_seed,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            epochs=args.epochs,
            warmup_fraction=args.warmup_fraction,
            decay_fraction=args.decay_fraction,
            decay_transitions=args.decay_transitions,
            decay_factor=args.decay_factor,
            l2_loss_weight=args.l2_loss_weight,
            dilation_iterations=args.dilation_iterations,
            max_distance=args.max_distance,
            temperature=args.temperature,
            epsilon=args.epsilon,
            checkpoint_every=args.checkpoint_every,
            device=args.device
        )

    elif args.command == 'predict':

        input_path = Path(args.input_path)
        output_path = Path(args.output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Get a list of image paths.
        image_paths = []
        if input_path.is_dir():
            for path in input_path.glob("*"):
                if path.is_file():
                    image_paths.append(path)
        else:
            image_paths.append(input_path)

        # Create the Piscis model.
        model = Piscis(model_name=args.model_name, batch_size=args.batch_size, input_size=None)

        # Loop over image paths.
        for image_path in tqdm(image_paths):

            # Load an image and predict spots.
            image = iio.imread(image_path)
            coords = model.predict(
                image,
                stack=args.stack,
                scale=args.scale,
                threshold=args.threshold,
                min_distance=args.min_distance,
                intermediates=False
            )

            # Save the predicted spots to a CSV file.
            np.savetxt(output_path / f'{image_path.stem}.csv', coords, delimiter=',')

    else:

        parser.print_help()


if __name__ == '__main__':
    main()
