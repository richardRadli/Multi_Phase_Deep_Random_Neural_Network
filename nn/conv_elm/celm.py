import torch
import torch.nn as nn
import torchvision.datasets as datasets

from torch.utils.data import DataLoader
from tqdm import tqdm
from torchsummary import summary
from torchvision.transforms import transforms


class Net(nn.Module):
    def __init__(self, classes, image_size, channel_list, in_channels=1):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, channel_list[0], 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(channel_list[0], channel_list[1], 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(channel_list[1], channel_list[2], 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(channel_list[2], channel_list[3], 3, padding=1),
            nn.ReLU(),

            nn.Flatten()
        )

        dummy_input = torch.randn(1, in_channels, image_size, image_size)
        with torch.no_grad():
            flatten_output = self.features(dummy_input)

        # Get the number of features after Flatten
        flatten_size = flatten_output.size(1)
        self.head = nn.Linear(flatten_size, classes)

        self.set_bias()

    def forward(self, x):
        out = self.features(x)
        out = self.head(out)
        return out

    def set_bias(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=0.01)


def create_matrices(model, loader, classes, device):
    HH = 0.
    HY = 0.
    with torch.no_grad():
        for X, Y in tqdm(loader):
            X, Y = X.to(device), Y.to(device)

            features = model.features(X)
            ones = torch.ones(X.size(0), 1, device=device)
            features = torch.cat((features, ones), dim=1)

            HH = HH + features.T @ features
            Y_one_hot = torch.nn.functional.one_hot(Y, classes)
            HY = HY + features.T @ Y_one_hot.float()

    return HH, HY


def solve_linear(HH, HY, device, alpha=0.):
    n = HH.size(0)
    HH_reg = HH + alpha * torch.eye(n, device=device)  # add regularization
    sol = torch.linalg.lstsq(HH_reg, HY)
    beta_weights = sol.solution
    return beta_weights


def update_model(model, beta_weights):
    model.head.weight.data = beta_weights[:-1].T
    model.head.bias.data = beta_weights[-1]


def train(model, loader, classes, device, alpha=0.):
    HH, HY = create_matrices(model, loader, classes, device)
    B = solve_linear(HH, HY, device, alpha)
    update_model(model, B)


def evaluate(model, loader, device):
    num_correct = 0
    num_total = 0
    with torch.no_grad():
        for test_x, test_y in loader:
            test_x, test_y = test_x.to(device), test_y.to(device)
            out = model(test_x)
            pred_labels = torch.argmax(out, dim=1)
            num_correct += (pred_labels == test_y).sum().item()
            num_total += test_x.size(0)

    acc = num_correct / num_total
    return acc


def main():
    torch.manual_seed(42)
    train_dset = (
        datasets.MNIST(root="C:/Users/ricsi/Desktop", train=True, download=False, transform=transforms.ToTensor())
    )

    test_dset = (
        datasets.MNIST(root="C:/Users/ricsi/Desktop", train=False, download=False, transform=transforms.ToTensor())
    )

    train_loader = DataLoader(train_dset, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_dset, batch_size=256, shuffle=False)

    model = Net(10, 28, channel_list=[128, 128, 256, 512]).to("cuda")
    summary(model, (1, 28, 28))

    train(model, train_loader, 10, "cuda", 1e-5)

    test_acc = evaluate(model, test_loader, "cuda")
    print(test_acc)


if __name__ == "__main__":
    main()
