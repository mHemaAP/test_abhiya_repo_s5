import os
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
from torchvision import datasets, transforms
from torch.utils.data import DataLoader 
import torch.multiprocessing as mp
from model import Net
import time

def train_epoch(epoch, args, model, train_loader, optimizer):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to('cpu'), target.to('cpu')
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

def train(rank, args, model, dataloader_kwargs):
    torch.manual_seed(args.seed + rank)
    # Create DataLoader
    train_loader = DataLoader(datasets.MNIST('/opt/mount', train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])), **dataloader_kwargs)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    for epoch in range(1, args.epochs + 1):
        train_epoch(epoch, args, model, train_loader, optimizer)

def main(): 

    save_dir = "/opt/mount" 
    
    parser = argparse.ArgumentParser(description="MNIST Training Script")
    parser.add_argument("--batch-size", type=int, default=64, metavar="N", help="input batch size for training (default: 64)")
    parser.add_argument("--lr", type=float, default=0.01, metavar="LR", help="learning rate (default: 0.01)")
    parser.add_argument("--momentum", type=float, default=0.5, metavar="M", help="SGD momentum (default: 0.5)")
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    parser.add_argument("--log-interval", type=int, default=10, metavar="N", help="how many batches to wait before logging training status")
    parser.add_argument("--num-processes", type=int, default=2, metavar="N", help="how many training processes to use (default: 2)")
    parser.add_argument("--dry-run", action="store_true", default=False, help="quickly check a single pass")
    parser.add_argument("--save_dir", default="/opt/mount", help="checkpoint will be saved in this directory")
    parser.add_argument('--epochs', type=int, default=1, metavar='N', help='number of epochs to train (default: 1)')
    parser.add_argument('--save_model', action='store_true', default=True, help='save the trained model to state_dict')

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    mp.set_start_method("spawn", force=True)

    model = Net()
    model.share_memory()

    kwargs = {
        "batch_size": args.batch_size,
        "num_workers": 1,
        "pin_memory": True,
        "shuffle": True,
    }

    processes = []
    for rank in range(args.num_processes):
        p = mp.Process(target=train, args=(rank, args, model, kwargs))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    

    if args.save_model:
        os.makedirs(os.path.join(args.save_dir, "model"), exist_ok=True)
        torch.save(model.state_dict(), os.path.join(args.save_dir, "model", "mnist_cnn.pt"))

    # time.sleep(10000)
        #/opt/mount/model/mnist.pt

if __name__ == "__main__":
    main()
