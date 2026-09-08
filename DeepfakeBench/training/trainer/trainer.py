"""Training and validation loop used by Effort++."""

import os
import pickle
from collections import defaultdict

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from metrics.base_metrics_class import Recorder
from metrics.utils import get_test_metrics


class Trainer:
    def __init__(self, config, model, optimizer, logger, metric_scoring="auc"):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.logger = logger
        self.metric_scoring = metric_scoring
        self.latest_validation_loss = None
        self.best_metrics_all_time = defaultdict(
            lambda: defaultdict(
                lambda: float("inf") if metric_scoring == "eer" else float("-inf")
            )
        )
        self.writers = {}
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        self.model.device = self.device


        self.log_dir = os.path.join(config["log_dir"], config["model_name"])
        os.makedirs(self.log_dir, exist_ok=True)

    @property
    def detector(self):
        return self.model

    def get_writer(self, phase, dataset, metric):
        key = f"{phase}-{dataset}-{metric}"
        if key not in self.writers:
            path = os.path.join(self.log_dir, phase, dataset, metric)
            self.writers[key] = SummaryWriter(path)
        return self.writers[key]

    def set_train(self):
        self.model.train()

    def set_eval(self):
        self.model.eval()

    def save_checkpoint(self, dataset, details):
        path = os.path.join(self.log_dir, "test", dataset)
        os.makedirs(path, exist_ok=True)
        torch.save(self.detector.state_dict(), os.path.join(path, "ckpt_best.pth"))
        self.logger.info("Checkpoint saved for %s (%s)", dataset, details)

    def save_metrics(self, dataset, values):
        path = os.path.join(self.log_dir, "test", dataset)
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "metric_dict_best.pickle"), "wb") as handle:
            pickle.dump(values, handle)

    def train_step(self, data):
        predictions = self.model(data)
        losses = self.detector.get_losses(data, predictions)
        self.optimizer.zero_grad(set_to_none=True)
        losses["overall"].backward()
        self.optimizer.step()
        return losses, predictions

    def train_epoch(self, epoch, train_data_loader, test_data_loaders=None):
        loss_recorders = defaultdict(Recorder)
        metric_recorders = defaultdict(Recorder)
        self.set_train()

        for step, data in tqdm(enumerate(train_data_loader), total=len(train_data_loader)):
            for key, value in data.items():
                if torch.is_tensor(value):
                    data[key] = value.to(self.device, non_blocking=True)

            losses, predictions = self.train_step(data)
            metrics = self.detector.get_train_metrics(data, predictions)
            for name, value in losses.items():
                loss_recorders[name].update(float(value.detach()))
            for name, value in metrics.items():
                metric_recorders[name].update(value)

            if step % int(self.config.get("rec_iter", 100)) == 0:
                for name, recorder in loss_recorders.items():
                    average = recorder.average()
                    self.get_writer("train", "train", name).add_scalar(
                        f"loss/{name}", average, epoch * len(train_data_loader) + step
                    )
                loss_recorders.clear()
                metric_recorders.clear()

        if test_data_loaders:
            return self.test_epoch(epoch, test_data_loaders)
        return None

    @torch.no_grad()
    def inference(self, data):
        return self.model(data, inference=True)

    def test_one_dataset(self, loader):
        losses = defaultdict(Recorder)
        probabilities = []
        labels = []
        for data in tqdm(loader, total=len(loader)):
            data["label"] = torch.where(data["label"] != 0, 1, 0)
            for key, value in data.items():
                if torch.is_tensor(value):
                    data[key] = value.to(self.device, non_blocking=True)
            predictions = self.inference(data)
            batch_losses = self.detector.get_losses(data, predictions)
            for name, value in batch_losses.items():
                losses[name].update(float(value.detach()))
            probabilities.extend(predictions["prob"].cpu().numpy())
            labels.extend(data["label"].cpu().numpy())
        return losses, np.asarray(probabilities), np.asarray(labels)

    def is_improved(self, dataset, metrics):
        old = self.best_metrics_all_time[dataset][self.metric_scoring]
        new = metrics[self.metric_scoring]
        return new < old if self.metric_scoring == "eer" else new > old

    def test_epoch(self, epoch, test_data_loaders):
        self.set_eval()
        validation_losses = []
        for dataset_name, loader in test_data_loaders.items():
            losses, probabilities, labels = self.test_one_dataset(loader)
            overall = losses["overall"].average()
            if overall is not None:
                validation_losses.append(float(overall))

            metrics = get_test_metrics(
                y_pred=probabilities,
                y_true=labels,
                img_names=loader.dataset.data_dict["image"],
            )
            if self.is_improved(dataset_name, metrics):
                self.best_metrics_all_time[dataset_name].update(metrics)
                if self.config.get("save_ckpt", True):
                    self.save_checkpoint(dataset_name, f"epoch={epoch}")
                self.save_metrics(dataset_name, metrics)

            for name, value in metrics.items():
                if name not in {"pred", "label", "dataset_dict"}:
                    self.get_writer("validation", dataset_name, name).add_scalar(
                        f"metric/{name}", value, epoch
                    )
            self.logger.info("%s metrics: %s", dataset_name, metrics)

        self.latest_validation_loss = (
            float(np.mean(validation_losses)) if validation_losses else None
        )
        self.set_train()
        return self.best_metrics_all_time
