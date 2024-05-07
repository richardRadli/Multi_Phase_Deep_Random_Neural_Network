import torch
import torch.nn as nn
import torchvision

from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm


class CELM(nn.Module):
    def __init__(self):
        super(CELM, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 32 * 32, 1024)
        self.weights = nn.Parameter(torch.randn(1024, 1000))

    def forward(self, x):
        x = self.conv1(x)
        x = torch.nn.LeakyReLU(negative_slope=0.2)(x)
        x = self.conv2(x)
        x = torch.nn.LeakyReLU(negative_slope=0.2)(x)
        x = x.view(-1, 64 * 32 * 32)
        x = self.fc1(x)
        x = torch.nn.LeakyReLU(negative_slope=0.2)(x)
        x = torch.nn.LeakyReLU(negative_slope=0.2)(x @ self.weights)
        return x


def main():
    torch.manual_seed(42)
    model = CELM().to("cpu")
    transform = transforms.Compose([transforms.ToTensor()])
    train_data = torchvision.datasets.CIFAR10(
        root='C:/Users/ricsi/Desktop', train=True, download=True, transform=transform
    )
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=50000, shuffle=False)

    test_data = torchvision.datasets.CIFAR10(
        root='C:/Users/ricsi/Desktop', train=False, download=True, transform=transform
    )
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=10000, shuffle=False)

    for train_x, train_y in tqdm(train_loader, total=len(train_loader)):
        train_x = train_x.to("cpu")
        train_y = train_y.to("cpu")
        train_y = torch.nn.functional.one_hot(train_y).float()

        model(train_x)
        with torch.no_grad():
            h1 = model(train_x)
            beta = torch.pinverse(h1).matmul(train_y)

    for test_x, test_y in tqdm(test_loader, total=len(test_loader)):
        test_x = test_x.to("cpu")
        test_y = test_y.to("cpu")
        beta_weights = beta.to("cpu")
        test_y = torch.nn.functional.one_hot(test_y).float()
        with torch.no_grad():
            h1 = model(test_x)
            predictions = h1.matmul(beta_weights)
            y_predicted_argmax = torch.argmax(predictions, dim=-1)
            y_true_argmax = torch.argmax(test_y, dim=-1)
        accuracy = accuracy_score(y_true_argmax.detach().cpu(), y_predicted_argmax.detach().cpu())

        print(accuracy)


if __name__ == '__main__':
    main()
