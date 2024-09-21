import os
import json
import torch
import torch.nn.functional as F
import argparse
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.nn import Conv2d, MaxPool2d, Linear, Sequential, ReLU, LogSoftmax, Flatten

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.main = Sequential(
            Conv2d(1, 32, 3, 1),
            ReLU(),
            Conv2d(32, 64, 3, 1),
            ReLU(),
            MaxPool2d(2),
            Flatten(),
            Linear(9216, 128),
            ReLU(),
            Linear(128, 10),
            LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.main(x)

def test_epoch(model, device, data_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            output = model(data.to(device))
            test_loss += F.nll_loss(output, target.to(device), reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.to(device).view_as(pred)).sum().item()

    test_loss /= len(data_loader.dataset)
    accuracy = 100. * correct / len(data_loader.dataset)
    out = {'Test loss': test_loss, 'Accuracy': accuracy}
    print(out)
    return out


def main():
    parser = argparse.ArgumentParser(description='MNIST Evaluation Script')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    args = parser.parse_args()
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        kwargs.update({'num_workers': 1, 'pin_memory': True, 'shuffle': True},)
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    test_loader = DataLoader(test_dataset, **kwargs)
    model = Net().to(device)
    model.load_state_dict(torch.load("./model/mnist_cnn.pt"))
    eval_results = test_epoch(model, device, test_loader)

    with open('./model/eval_results.json', 'w') as f:
        json.dump(eval_results, f)

if __name__ == "__main__":
    main()
