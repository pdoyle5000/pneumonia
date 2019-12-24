import sys
from typing import Optional
import numpy as np
import torch
from dataset import PneumoniaDataset, SetType
from simple_net import SimpleNet
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix
import torch.optim as optim

from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn


class PneumoniaTrainer:
    def __init__(
        self, model_name: str, epochs: int = 100, config: Optional[dict] = None
    ):
        self.output_model = model_name + ".pth"
        self.train_set = PneumoniaDataset(SetType.train)
        self.train_loader = DataLoader(
            PneumoniaDataset(SetType.train), batch_size=16, shuffle=True, num_workers=8
        )
        self.val_loader = DataLoader(
            PneumoniaDataset(SetType.val, shuffle=False),
            batch_size=16,
            shuffle=False,
            num_workers=8,
        )
        self.test_loader = DataLoader(
            PneumoniaDataset(SetType.test, shuffle=False),
            batch_size=16,
            shuffle=False,
            num_workers=8,
        )
        self.config = {
            "pos_weight_bias": 0.5,
            "starting_lr": 1e-2,
            "momentum": 0.9,
            "decay": 5e-4,
            "lr_adjustment_factor": 0.3,
            "scheduler_patience": 15,
            "print_cadence": 100,
            "comment": "Added large FC layer.",
            "pos_weight": 1341 / 3875,  # Number of negatives / positives.
        }

        self.epochs = epochs
        self.device = torch.device("cuda:0")
        self.writer = SummaryWriter(comment=self.config["comment"])
        self.net = SimpleNet(1).to(self.device)
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(self.config["pos_weight"])
        )
        self.optimizer = optim.SGD(
            self.net.parameters(),
            lr=self.config["starting_lr"],
            momentum=self.config["momentum"],
            weight_decay=self.config["decay"],
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            factor=self.config["lr_adjustment_factor"],
            mode="max",
            verbose=True,
            patience=self.config["scheduler_patience"],
        )

        print("Trainer Initialized.")
        for dataset in [self.train_loader, self.test_loader, self.val_loader]:
            print(f"Size of set: {len(dataset)}")

    def train(self):
        training_pass = 0
        for epoch in range(self.epochs):
            running_loss = 0.0
            for i, (inputs, labels, metadata) in enumerate(self.train_loader):
                self.net.train()
                self.optimizer.zero_grad()
                outputs = self.net(inputs.float().to(self.device))
                loss = self.criterion(
                    outputs, labels.unsqueeze(1).float().to(self.device)
                )
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                if i > 0 and i % self.config["print_cadence"] == 0:
                    print(
                        f'Epoch: {epoch}\tBatch: {i}\tLoss: {running_loss / self.config["print_cadence"]}'
                    )
                    self.writer.add_scalar(
                        "Train/RunningLoss",
                        running_loss / self.config["print_cadence"],
                        training_pass,
                    )
                    running_loss = 0.0
                training_pass += 1
            train_accuracy = self.log_training_metrics(epoch)
            self.log_validation_metrics(epoch)
            self.scheduler.step(train_accuracy)
        accuracy, metrics = self.calculate_accuracy(self.test_loader)
        self.writer.add_text("Test/Accuracy", f"{accuracy}")
        for key, val in metrics.items():
            self.writer.add_text(f"Test/{key}", f"{val}")
        self.save_model()

    def log_training_metrics(self, epoch: int):
        accuracy, metrics = self.calculate_accuracy(self.train_loader)
        self.writer.add_scalar(f"Train/Accuracy", accuracy, epoch)
        for key, val in metrics.items():
            self.writer.add_scalar(f"Train/{key}", val, epoch)
        return accuracy

    def log_validation_metrics(self, epoch: int):
        accuracy, metrics = self.calculate_accuracy(self.val_loader)
        self.writer.add_scalar("Validation/Accuracy", accuracy, epoch)
        for key, val in metrics.items():
            self.writer.add_scalar(f"Validation/{key}", val, epoch)
        return accuracy

    def calculate_accuracy(self, loader: DataLoader):
        truth_list = []
        pred_list = []
        with torch.no_grad():
            self.net.eval()
            correct = 0.0
            total = 0.0
            for inputs, labels, metadata in loader:
                outputs = self.net(inputs.float().to(self.device))
                sigmoid = torch.nn.Sigmoid()
                preds = sigmoid(outputs)
                preds = np.round(preds.detach().cpu().squeeze(1))
                pred_list.extend(preds)
                truth_list.extend(labels)
                total += labels.size(0)
                correct += preds.eq(labels.float()).sum().item()
        print(f"Correct:\t{correct}, Incorrect:\t{total-correct}")

        tn, fp, fn, tp = confusion_matrix(truth_list, pred_list).ravel()
        metrics = {
            "Recall": tp / (tp + fn),
            "Precision": tp / (tp + fp),
            "FalseNegativeRate": fn / (tn + fn),
            "FalsePositiveRate": fp / (tp + fp),
        }

        return correct / total, metrics

    def save_model(self):
        print("saving...")
        torch.save(self.net.state_dict(), self.output_model)


if __name__ == "__main__":
    trainer = PneumoniaTrainer(sys.argv[1], 300)
    trainer.train()
