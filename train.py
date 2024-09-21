
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import argparse
import os

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Define model architecture
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)  # 1 input channel (grayscale), 10 output channels, 5x5 kernel
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5) # 10 input channels, 20 output channels
        self.conv2_drop = nn.Dropout2d()              # Dropout layer for regularization
        self.fc1 = nn.Linear(320, 50)                 # Fully connected layer
        self.fc2 = nn.Linear(50, 10)                  # Output layer (10 classes for MNIST)

    def forward(self, x):
        # Define the forward pass
        x = F.relu(F.max_pool2d(self.conv1(x), 2))              # Conv -> ReLU -> Max Pool
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))  # Conv -> Dropout -> ReLU -> Max Pool
        x = x.view(-1, 320)                                     # Flatten the tensor
        x = F.relu(self.fc1(x))                                 # Fully connected -> ReLU
        x = F.dropout(x, training=self.training)                # Apply dropout during training
        x = self.fc2(x)                                         # Final fully connected layer
        return F.log_softmax(x, dim=1)                          # Log Softmax for classification

def train_epoch(epoch, args, model, device, data_loader, optimizer):
    model.train()  # Set the model to training mode
    pid = os.getpid()  # Get process ID for logging
    for batch_idx, (data, target) in enumerate(data_loader):
        optimizer.zero_grad()                # Zero out gradients
        output = model(data.to(device))      # Forward pass
        loss = F.nll_loss(output, target.to(device))  # Calculate loss
        loss.backward()                      # Backpropagate
        optimizer.step()                     # Update the model's parameters

        # Log the training status
        if batch_idx % args.log_interval == 0:
            print(f'{pid}\\tTrain Epoch: {epoch} [{batch_idx * len(data)}/{len(data_loader.dataset)} ({100. * batch_idx / len(data_loader):.0f}%)]\\tLoss: {loss.item():.6f}')
            if args.dry_run:
                break

def test_epoch(model, device, data_loader):
    model.eval()  # Set the model to evaluation mode
    test_loss = 0
    correct = 0
    with torch.no_grad():  # Disable gradient calculation
        for data, target in data_loader:
            output = model(data.to(device))  # Forward pass
            test_loss += F.nll_loss(output, target.to(device), reduction='sum').item()  # Sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # Get the index of the max log-probability
            correct += pred.eq(target.to(device).view_as(pred)).sum().item()  # Count correct predictions

    test_loss /= len(data_loader.dataset)  # Average loss
    accuracy = 100. * correct / len(data_loader.dataset)
    print(f'\\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(data_loader.dataset)} ({100. * correct / len(data_loader.dataset):.0f}%)\\n')
    return accuracy


def main():
    # Parser to get command line arguments
    parser = argparse.ArgumentParser(description='MNIST Training Script')

    # Define command line arguments
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=3, metavar='N',              #REDUCING EPOCHS TO SAVE TIME
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    # parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
    #                     help='SGD momentum (default: 0.5)')                         CAN LATER CHNAGE TO SGD IF REQUIRED
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='resume training from checkpoint')
    
    #args = parser.parse_args()
    args, unknown = parser.parse_known_args()
    
    use_cuda = args.cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    # Load the MNIST dataset for training and testing
    train_data = torchvision.datasets.MNIST(root="./datasets", transform=transforms.ToTensor(), train=True, download=True)
    test_data = torchvision.datasets.MNIST(root="./datasets", transform=transforms.ToTensor(), train=False, download=True)
    
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=args.test_batch_size, shuffle=False)

    model = Net().to(device)
    
    # Optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile('mnist_checkpoint.pth'):
            print("=> Loading checkpoint...")
            model.load_state_dict(torch.load('mnist_checkpoint.pth'))

    # Define the optimizer
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)



    # Training and testing cycles
    best_accuracy = 0  # Initialize best accuracy tracker

    for epoch in range(1, args.epochs + 1):
        train_epoch(epoch, args, model, device, train_loader, optimizer)
        accuracy = test_epoch(model, device, test_loader)
        
        # Save the model if it's the best one so far
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), 'model_checkpoint.pth')
            # model_save_path = os.path.join(os.getcwd(), 'model_checkpoint.pth')
            # torch.save(model.state_dict(), model_save_path)
            print(f'Best model saved with accuracy: {best_accuracy:.2f}%')


if __name__ == "__main__":
    main()
