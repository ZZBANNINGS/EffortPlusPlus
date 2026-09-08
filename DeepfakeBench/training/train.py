"""Training entry point for the paper Effort++ recipe."""

import argparse
import datetime
import os
import random

import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import yaml

from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
from detectors import DETECTOR
from logger import create_logger
from metrics.utils import parse_metric_for_print
from trainer.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train Effort++")
    parser.add_argument(
        "--detector_path",
        default="./training/config/detector/effort_svd_lu_slerp_cos.yaml",
    )
    parser.add_argument("--train_dataset", nargs="+")
    parser.add_argument("--validation_dataset", nargs="+")
    parser.add_argument("--no-save_ckpt", dest="save_ckpt", action="store_false", default=True)
    return parser.parse_args()


def init_seed(config):
    seed = config.get("manualSeed")
    if seed is None:
        seed = random.randint(1, 10000)
        config["manualSeed"] = seed
    random.seed(seed)
    torch.manual_seed(seed)
    if config.get("cuda", True) and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_training_data(config):
    dataset = DeepfakeAbstractBaseDataset(config=config, mode="train")
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=config["train_batchSize"],
        shuffle=True,
        num_workers=int(config["workers"]),
        collate_fn=dataset.collate_fn,
    )


def prepare_validation_data(config):
    loaders = {}
    for dataset_name in config["validation_dataset"]:
        dataset_config = config.copy()
        dataset_config["validation_dataset"] = dataset_name
        dataset = DeepfakeAbstractBaseDataset(config=dataset_config, mode="val")
        loaders[dataset_name] = torch.utils.data.DataLoader(
            dataset,
            batch_size=config["test_batchSize"],
            shuffle=False,
            num_workers=int(config["workers"]),
            collate_fn=dataset.collate_fn,
        )
    return loaders


def choose_optimizer(model, config):
    if config["optimizer"]["type"] != "adam":
        raise ValueError("The paper recipe uses Adam")
    options = config["optimizer"]["adam"]
    return optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=options["lr"],
        betas=(options["beta1"], options["beta2"]),
        eps=options["eps"],
        weight_decay=options["weight_decay"],
        amsgrad=False,
    )


def main():
    args = parse_args()
    with open(args.detector_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    with open("./training/config/train_config.yaml", "r", encoding="utf-8") as handle:
        config.update(yaml.safe_load(handle))

    if args.train_dataset:
        config["train_dataset"] = args.train_dataset
    if args.validation_dataset:
        config["validation_dataset"] = args.validation_dataset
    config.update(
        {
            "save_ckpt": args.save_ckpt,
        }
    )

    if config.get("cuda", True) and not torch.cuda.is_available():
        raise RuntimeError("Effort++ training requires a CUDA-capable GPU")
    torch.cuda.set_device(0)

    os.makedirs(config["log_dir"], exist_ok=True)
    logger = create_logger(os.path.join(config["log_dir"], "training.log"))
    logger.info("Configuration:\n%s", yaml.safe_dump(config, sort_keys=False))

    init_seed(config)
    cudnn.benchmark = bool(config.get("cudnn", True))


    train_loader = prepare_training_data(config)
    validation_loaders = prepare_validation_data(config)
    model = DETECTOR[config["model_name"]](config)
    optimizer = choose_optimizer(model, config)
    trainer = Trainer(config, model, optimizer, logger, config["metric_scoring"])

    early_stopping = bool(config.get("early_stopping", True))
    patience = int(config.get("early_stopping_patience", 10))
    best_validation_loss = float("inf")
    stale_epochs = 0
    best_metric = None

    for epoch in range(int(config.get("start_epoch", 1)), int(config["nEpochs"]) + 1):
        best_metric = trainer.train_epoch(epoch, train_loader, validation_loaders)
        logger.info(
            "Epoch %d validation: %s",
            epoch,
            parse_metric_for_print(best_metric),
        )

        validation_loss = trainer.latest_validation_loss
        if early_stopping and validation_loss is not None:
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                stale_epochs = 0
            else:
                stale_epochs += 1
                logger.info("Early stopping patience: %d/%d", stale_epochs, patience)
            if stale_epochs >= patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

    for writer in trainer.writers.values():
        writer.close()


if __name__ == "__main__":
    main()
