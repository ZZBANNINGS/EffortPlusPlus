"""Evaluate an Effort++ checkpoint on one or more paper datasets."""

import argparse
import random

import numpy as np
import torch
import yaml
from tqdm import tqdm

from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
from detectors import DETECTOR
from metrics.utils import get_test_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Effort++")
    parser.add_argument(
        "--detector_path",
        default="./training/config/detector/effort_svd_lu_slerp_cos.yaml",
    )
    parser.add_argument("--test_dataset", nargs="+")
    parser.add_argument("--weights_path", required=True)
    return parser.parse_args()


def prepare_loaders(config):
    loaders = {}
    for dataset_name in config["test_dataset"]:
        dataset_config = config.copy()
        dataset_config["test_dataset"] = dataset_name
        dataset = DeepfakeAbstractBaseDataset(config=dataset_config, mode="test")
        loaders[dataset_name] = torch.utils.data.DataLoader(
            dataset,
            batch_size=config["test_batchSize"],
            shuffle=False,
            num_workers=int(config["workers"]),
            collate_fn=dataset.collate_fn,
        )
    return loaders


def load_checkpoint(model, path):
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint.get("state_dict", checkpoint)
    state = {
        key[7:] if key.startswith("module.") else key: value
        for key, value in state.items()
    }
    model.load_state_dict(state, strict=True)


@torch.inference_mode()
def evaluate(model, loader, device):
    probabilities = []
    labels = []
    for data in tqdm(loader, total=len(loader)):
        data["label"] = torch.where(data["label"] != 0, 1, 0)
        for key, value in data.items():
            if torch.is_tensor(value):
                data[key] = value.to(device, non_blocking=True)
        predictions = model(data, inference=True)
        probabilities.extend(predictions["prob"].cpu().numpy())
        labels.extend(data["label"].cpu().numpy())

    return get_test_metrics(
        y_pred=np.asarray(probabilities),
        y_true=np.asarray(labels),
        img_names=loader.dataset.data_dict["image"],
    )


def main():
    args = parse_args()
    with open(args.detector_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    with open("./training/config/test_config.yaml", "r", encoding="utf-8") as handle:
        config.update(yaml.safe_load(handle))
    if args.test_dataset:
        config["test_dataset"] = args.test_dataset

    seed = int(config.get("manualSeed", 1024))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DETECTOR[config["model_name"]](config)
    load_checkpoint(model, args.weights_path)
    model.to(device).eval()

    for dataset_name, loader in prepare_loaders(config).items():
        print(f"{dataset_name}: {evaluate(model, loader, device)}")


if __name__ == "__main__":
    main()
