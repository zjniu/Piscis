import argparse
import imageio.v3 as iio
import numpy as np

from pathlib import Path
from tqdm.auto import tqdm

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
    predict_parser.add_argument('input_path', type=str, help="Path to image or stack of images.")
    predict_parser.add_argument('output_path', type=str, help="Path to save predicted spots as CSV.")
    predict_parser.add_argument('--model-name', type=str, default='20230709', help="Model name.")
    predict_parser.add_argument('--batch-size', type=int, default=4, help="Batch size for the CNN.")
    predict_parser.add_argument('--stack', action='store_true', help="Whether the input is a stack of images.")
    predict_parser.add_argument('--scale', type=float, default=1.0, help="Scale factor for rescaling the input.")
    predict_parser.add_argument('--threshold', type=float, default=2.0,
                                help="Spot detection threshold. Can be interpreted as the minimum number of fully "
                                     "confident pixels necessary to identify a spot. Default is 2.0."
                                )
    predict_parser.add_argument('--min-distance', type=int, default=1, help="Minimum distance between spots.")

    # Train subparser.
    train_parser = subparsers.add_parser('train', formatter_class=formatter_class, help="Train a SpotsModel")
    train_parser.add_argument('model_name', type=str, help="Name of a new or existing model.")
    train_parser.add_argument('dataset_path', type=str,
                              help="Path to the directory containing training and validation datasets.")
    train_parser.add_argument('--adjustment', type=str, default='standardize',
                              help="Adjustment type applied to images. Supported types are 'normalize' and "
                                   "'standardize'.")
    train_parser.add_argument('--input-size', type=int, nargs=2, default=(256, 256),
                              help="Size of the input images used for training.")
    train_parser.add_argument('--random-seed', type=int, default=0,
                              help="Random seed used for initialization and training.")
    train_parser.add_argument('--batch-size', type=int, default=4, help="Batch size for training.")
    train_parser.add_argument('--learning-rate', type=float, default=0.05, help="Learning rate for the optimizer.")
    train_parser.add_argument('--epochs', type=int, default=200, help="Number of epochs to train the model for.")
    train_parser.add_argument('--warmup-epochs', type=int, default=10,
                              help="Number of warmup epochs for learning rate scheduling.")
    train_parser.add_argument('--decay-epochs', type=int, default=100,
                              help="Number of decay epochs for learning rate scheduling.")
    train_parser.add_argument('--decay-rate', type=float, default=0.5, help="Decay rate for learning rate scheduling.")
    train_parser.add_argument('--decay-transition-epochs', type=int, default=10,
                              help="Number of epochs for each decay transition in learning rate scheduling.")
    train_parser.add_argument('--dilation-iterations', type=int, default=1,
                              help="Number of iterations to dilate ground truth labels to minimize class imbalance and "
                                   "misclassifications due to minor offsets.")
    train_parser.add_argument('--rmse-loss-weight', type=float, default=0.5, help="Weight for the rmse loss term.")
    train_parser.add_argument('--bce-loss-weight', type=float, default=0.0, help="Weight for the bce loss term.")
    train_parser.add_argument('--dice-loss-weight', type=float, default=0.0, help="Weight for the dice loss term.")
    train_parser.add_argument('--smoothf1-loss-weight', type=float, default=1.0,
                              help="Weight for the smoothf1 loss term.")

    args = parser.parse_args()

    if args.command == 'train':

        # Create the loss weights dictionary.
        loss_weights = {}
        if args.rmse_loss_weight > 0:
            loss_weights['rmse'] = args.rmse_loss_weight
        if args.bce_loss_weight > 0:
            loss_weights['bce'] = args.bce_loss_weight
        if args.dice_loss_weight > 0:
            loss_weights['dice'] = args.dice_loss_weight
        if args.smoothf1_loss_weight > 0:
            loss_weights['smoothf1'] = args.smoothf1_loss_weight

        # Train the model.
        train_model(
            model_name=args.model_name,
            dataset_path=args.dataset_path,
            adjustment=args.adjustment,
            input_size=args.input_size,
            random_seed=args.random_seed,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            warmup_epochs=args.warmup_epochs,
            decay_epochs=args.decay_epochs,
            decay_rate=args.decay_rate,
            decay_transition_epochs=args.decay_transition_epochs,
            dilation_iterations=args.dilation_iterations,
            loss_weights=loss_weights
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
