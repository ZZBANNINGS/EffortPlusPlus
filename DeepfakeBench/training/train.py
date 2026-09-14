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

    init_seed(config)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    run_dir = os.path.join(
        config["log_dir"],
        f"{config['model_name']}_{timestamp}",
    )
    os.makedirs(run_dir, exist_ok=False)
    config["run_dir"] = run_dir

    config_path = os.path.join(run_dir, "config.yaml")
    with open(config_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    logger = create_logger(os.path.join(run_dir, "training.log"))
    logger.info("Run directory: %s", run_dir)
    logger.info("Full configuration saved to %s", config_path)
    logger.info(
        "Training: model=%s, train=%s, validation=%s, batch=%d, workers=%d, "
        "epochs=%d, early_stopping=%s, patience=%d, seed=%s",
        config["model_name"],
        ",".join(config["train_dataset"]),
        ",".join(config["validation_dataset"]),
        config["train_batchSize"],
        config["workers"],
        config["nEpochs"],
        config.get("early_stopping", True),
        config.get("early_stopping_patience", 10),
        config.get("manualSeed"),
    )
    logger.info(
        "Method: SVD=%s (rank=%d), L2=%s, head=%s, uniformity_weight=%g, "
        "SLERP=%s (probability=%g)",
        config.get("use_svd", False),
        config.get("svd_trainable_rank", 1),
        config.get("use_l2_norm", False),
        config.get("head_type", "linear"),
        config.get("uniformity_weight", 0.0),
        config.get("slerp_enabled", False),
        config.get("slerp_probability", 0.0),
    )
    logger.info(
        "Paths: data=%s, manifests=%s, CLIP=%s",
        config["dataset_root_rgb"],
        config["dataset_json_folder"],
        config.get("clip_path", ""),
    )

    cudnn.benchmark = bool(config.get("cudnn", True))


    train_loader = prepare_training_data(config)
    validation_loaders = prepare_validation_data(config)
    model = DETECTOR[config["model_name"]](config)
    optimizer = choose_optimizer(model, config)
    trainer = Trainer(config, model, optimizer, logger, config["metric_scoring"])

    early_stopping = bool(config.get("early_stopping", True))
    patience = int(config.get("early_stopping_patience", 10))
    scoring = config["metric_scoring"]
    minimize_score = scoring == "eer"
    best_validation_score = float("inf") if minimize_score else float("-inf")
    stale_epochs = 0
    best_metric = None

    for epoch in range(int(config.get("start_epoch", 1)), int(config["nEpochs"]) + 1):
        best_metric = trainer.train_epoch(epoch, train_loader, validation_loaders)
        logger.info(
            "Epoch %d validation: %s",
            epoch,
            parse_metric_for_print(best_metric),
        )

        validation_score = trainer.best_validation_score_this_epoch
        if early_stopping and validation_score is not None:
            improved = (
                validation_score < best_validation_score
                if minimize_score
                else validation_score > best_validation_score
            )
            if improved:
                best_validation_score = validation_score
                stale_epochs = 0
            else:
                stale_epochs += 1
                logger.info(
                    "Early stopping patience: %d/%d (best %s=%.4f, current=%.4f)",
                    stale_epochs,
                    patience,
                    scoring,
                    best_validation_score,
                    validation_score,
                )
            if stale_epochs >= patience:
                logger.info(
                    "Early stopping at epoch %d (no %s improvement for %d epochs)",
                    epoch,
                    scoring,
                    patience,
                )
                break

    for writer in trainer.writers.values():
        writer.close()


if __name__ == "__main__":
    main()
