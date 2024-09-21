import json
import torch
import torch.nn as nn 
import torch.nn.functional as F
import argparse
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path
from model import Net 


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
    parser = argparse.ArgumentParser(description="MNIST Evaluation Script")

    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--save_dir", default="/opt/mount", help="checkpoint will be saved in this directory"
    )
    parser.add_argument(
        "--test-batch-size", type=int, default=1000, metavar="N", help="input batch size for testing (default: 1000)"
    )
    parser.add_argument(
        "--model-checkpoint", default="/opt/mount/model/mnist_cnn.pt", help="path to the saved model checkpoint"
    )

    args, unknown = parser.parse_known_args()
    use_cuda = torch.cuda.is_available()  
    torch.manual_seed(args.seed)
     
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        kwargs.update({'num_workers': 1, 'pin_memory': True, 'shuffle': True},)

    kwargs = {
        "batch_size": args.test_batch_size,
        "num_workers": 1,
        "pin_memory": True,
        "shuffle": True,
    }
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    # create MNIST test dataset and loader
    test_dataset = datasets.MNIST('/opt/mount', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # Load model
    model = Net()
    checkpoint_path = "/opt/mount/model/mnist_cnn.pt" 
    model.load_state_dict(torch.load(checkpoint_path))

    # Run the test epoch and collect evaluation results
    eval_results = test_epoch(model, device, test_loader)

    # # Save evaluation results to a JSON file
    # results_path = Path(args.save_dir) / "eval_results.json"

    # Save evaluation metrics
    with (Path(args.save_dir) / "model" / "eval_results.json").open("w") as f:
        json.dump(eval_results, f)


if __name__ == "__main__":
    main()
