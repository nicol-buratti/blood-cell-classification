from pathlib import Path

from data_loader.blood_cell_data_loader import BloodCellDataLoader
from models.blood_cell_model import ConvBloodCellModel
from trainers.blood_cell_trainer import ConvBloodCellTrainer
from utils.config import process_config

from utils.dirs import create_dirs
from utils.result_plotter import (
    plot_accuracy,
    plot_loss,
    plot_training_history,
    save_confusion_matrix,
    save_model_file,
)
from utils.utils import get_args

from tensorflow.random import set_seed
from kagglehub import dataset_download
import shutil


def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)
    except:
        print("missing or invalid arguments")
        exit(0)

    # Download dataset if it doesn't exist
    if not Path(config.data_loader.images_path).exists():
        print("Dataset not found, downloading...")
        dataset_path = dataset_download("paultimothymooney/blood-cells/version/6")
        shutil.move(
            Path(dataset_path) / "dataset2-master" / "dataset2-master" / "images",
            Path(config.data_loader.images_path),
        )

    # set global tensorflow seed
    set_seed(config.exp.seed)

    # create the experiments dirs
    create_dirs([config.callbacks.tensorboard_log_dir, config.callbacks.checkpoint_dir])

    print("Create the data generator.")
    data_loader = BloodCellDataLoader(config)

    print("Create the model.")
    model = ConvBloodCellModel(config)

    print("Create the trainer")
    trainer = ConvBloodCellTrainer(model.model, data_loader, config)

    print("Start training the model.")
    trainer.train()

    save_confusion_matrix(model.model, data_loader, config)
    plot_training_history(trainer, config)
    plot_loss(trainer, config)
    plot_accuracy(trainer, config)
    save_model_file(config)


if __name__ == "__main__":
    main()
