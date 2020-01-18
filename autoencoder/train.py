from torch import nn, device, save
from torch.optim import Adam, Adamax
from autoencoder import AutoEncoder
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset import EncoderDataset, SetType


class AutoEncoderTrainer:
    def __init__(self, epochs: int = 1000, output_name: str = "model"):
        self.epochs = epochs
        self.criterion = nn.MSELoss()
        self.device = device("cuda:0")
        self.model = AutoEncoder().to(self.device)
        self.output_path = f"{output_name}.pth"
        self.optimizer = Adam(self.model.parameters(), lr=1e-4)
        self.train_loader = DataLoader(
            EncoderDataset(SetType.val),
            batch_size=1, shuffle=True, num_workers=8)

        self.test_loader = DataLoader(
            EncoderDataset(SetType.test),
            batch_size=64, shuffle=True, num_workers=8)
        self.writer = SummaryWriter()

    def train(self):
        for epoch in range(self.epochs):
            running_loss = 0.0
            for i, (inputs, _, _) in enumerate(self.train_loader):
                self.model.train()
                self.optimizer.zero_grad()
                inputs = inputs.float().to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, inputs)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()*inputs.size(0)
            epoch_loss = running_loss / len(self.train_loader)
            print(f"Epoch: {epoch}\tLoss: {epoch_loss}")


if __name__ == "__main__":
    trainer = AutoEncoderTrainer()
    trainer.train()
    save(trainer.model.state_dict(), trainer.output_path)
