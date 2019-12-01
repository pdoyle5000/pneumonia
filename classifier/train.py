import sys
import numpy as np
import torch
from dataset import PneumoniaDataset, SetType
from simple_net import SimpleNet
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim

from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
import pickle


def main(model_name: str):
    output_model = "models/" + model_name + ".pth"
    writer = SummaryWriter()

    # Init datasets and loaders
    train_set = PneumoniaDataset(SetType.train)
    val_set = PneumoniaDataset(SetType.test, shuffle=False)  # for now.
    final_check_set = PneumoniaDataset(SetType.val, shuffle=False)  # for now.
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False, num_workers=8)
    final_loader = DataLoader(
        final_check_set, batch_size=16, shuffle=False, num_workers=8
    )
    for dataset in [train_set, val_set, final_check_set]:
        print(f"Size of {dataset.set_type} set: {len(dataset)}")

    # Init network, loss and optimizer
    net = SimpleNet(1).cuda()

    # There are twice as much pneumonia as healthy, offset the bias in the loss.
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(0.25))
    optimizer = optim.SGD(
        net.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4
    )
    scheduler = ReduceLROnPlateau(
        optimizer, factor=0.3, mode="max", verbose=True, patience=15
    )

    # Training Loop
    train_iter = 0
    for epoch in range(100):
        running_loss = 0.0
        for i, (inputs, labels, metadata) in enumerate(train_loader):
            net.train()
            optimizer.zero_grad()
            outputs = net(inputs.float().cuda())
            loss = criterion(outputs, labels.unsqueeze(1).float().cuda())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i > 0 and (i) % 100 == 0:
                print("[%d, %5d] loss %.3f" % (epoch, i, running_loss / 100))
                writer.add_scalar("TrainLoss", (running_loss / 100), train_iter)
                train_iter += 1
                running_loss = 0.0

        all_labels, all_preds, all_metadata, test_accuracy = calculate_accuracy(
            net, val_loader, "TestAcc", writer, epoch
        )
        scheduler.step(test_accuracy)
    calculate_accuracy(net, final_loader, "HoldoutAcc", writer, epoch)
    save_model_results(
        net, output_model, {"labels": all_labels, "preds": all_preds, "paths": all_data}
    )


def save_model_results(net, output_model, output_metadata):
    print("saving...")
    torch.save(net.state_dict(), output_model)
    pickle.dump(open(output_model + ".preds.pkl", "wb"), output_metadata)


def calculate_accuracy(net, loader, accuracy_label, writer, epoch):
    with torch.no_grad():
        net.eval()
        correct = 0.0
        total = 0.0
        i = 0.0
        all_labels = []
        all_preds = []
        all_data = []
        for inputs, labels, metadata in loader:
            outputs = net(inputs.float().cuda())
            sigmoid = torch.nn.Sigmoid()
            preds = sigmoid(outputs)
            preds = np.round(preds.cpu().squeeze(1))
            total += labels.size(0)
            correct += preds.eq(labels.float()).sum().item()
            i += 1
            all_labels.append(labels)
            all_preds.append(outputs)
            all_data.append(metadata)
        test_accuracy = correct / total
    print(f"Test Accuracy:\t{test_accuracy}")
    writer.add_scalar(accuracy_label, test_accuracy, epoch)
    print(f"Correct:\t{correct}, Incorrect:\t{total-correct}")
    return all_labels, all_preds, all_data, test_accuracy


if __name__ == "__main__":
    main(sys.argv[1])
