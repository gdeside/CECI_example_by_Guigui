#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import os
import multiprocessing
import ssl

# Fix SSL certificate issue (common on macOS)
ssl._create_default_https_context = ssl._create_unverified_context


# Define CNN architecture (must be at module level for multiprocessing)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


# Training function (must be at module level for multiprocessing)
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / len(loader), 100. * correct / total


# Evaluation function (must be at module level for multiprocessing)
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return running_loss / len(loader), 100. * correct / total


def main():
    print("=" * 60)
    print("CNN Training on CIFAR-10 - GPU vs CPU Demo")
    print("=" * 60)

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[INFO] Device Detection:")
    print(f"  - Device selected: {device}")

    if torch.cuda.is_available():
        print(f"  - GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"  - GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"  - CUDA Version: {torch.version.cuda}")
        print(f"  - Number of GPUs: {torch.cuda.device_count()}")
    else:
        print(f"  - No GPU detected, using CPU")
        num_cpus = multiprocessing.cpu_count()
        slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK', None)
        print(f"  - Available CPUs: {num_cpus}")
        if slurm_cpus:
            print(f"  - SLURM allocated CPUs: {slurm_cpus}")

    # Data preprocessing
    print(f"\n[STEP 1/5] Preparing data transformations...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load CIFAR-10 dataset
    print(f"\n[STEP 2/5] Loading CIFAR-10 dataset...")
    print(f"  - Checking for existing data in './data' directory...")

    # Check if data already exists
    data_exists = os.path.exists('./data/cifar-10-batches-py')
    if data_exists:
        print(f"  - Data found! Skipping download.")
    else:
        print(f"  - Data not found. Downloading (this may take a few minutes)...")

    try:
        train_dataset = datasets.CIFAR10(
            root='./data', train=True, download=not data_exists, transform=transform
        )
        test_dataset = datasets.CIFAR10(
            root='./data', train=False, download=not data_exists, transform=transform
        )
        print(f"  - Training samples: {len(train_dataset)}")
        print(f"  - Test samples: {len(test_dataset)}")
    except Exception as e:
        print(f"  - Error loading dataset: {e}")
        print(f"  - If download fails, manually download from:")
        print(f"    https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz")
        print(f"    and extract to './data' directory")
        exit(1)

    # Determine number of workers and pin_memory based on environment
    # Use 0 workers for local testing, 4 for HPC
    is_hpc = os.environ.get('SLURM_JOB_ID') is not None
    num_workers = 4 if is_hpc else 0
    use_pin_memory = torch.cuda.is_available()

    print(f"  - DataLoader workers: {num_workers} {'(HPC mode)' if is_hpc else '(local mode)'}")
    print(f"  - Pin memory: {use_pin_memory}")

    train_loader = DataLoader(
        train_dataset, batch_size=128, shuffle=True,
        num_workers=num_workers, pin_memory=use_pin_memory
    )
    test_loader = DataLoader(
        test_dataset, batch_size=128, shuffle=False,
        num_workers=num_workers, pin_memory=use_pin_memory
    )

    # Initialize model
    print(f"\n[STEP 3/5] Building CNN model...")
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  - Model architecture: SimpleCNN")
    print(f"  - Total parameters: {num_params:,}")
    print(f"  - Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"  - Model moved to: {device}")

    # Training loop
    print(f"\n[STEP 4/5] Training the model...")
    num_epochs = 20
    print(f"  - Number of epochs: {num_epochs}")
    print(f"  - Batch size: 128")
    print(f"  - Optimizer: Adam (lr=0.001)")
    print(f"  - Loss function: CrossEntropyLoss")
    print("-" * 60)

    start_time = time.time()
    best_test_acc = 0.0

    for epoch in range(num_epochs):
        epoch_start = time.time()

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        epoch_time = time.time() - epoch_start

        # Track best accuracy
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_marker = " *BEST*"
        else:
            best_marker = ""

        print(f"Epoch {epoch + 1:2d}/{num_epochs} ({epoch_time:5.1f}s) | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:5.2f}% | "
              f"Test Loss: {test_loss:.4f} Acc: {test_acc:5.2f}%{best_marker}")

    total_time = time.time() - start_time

    print("-" * 60)

    # Save results
    print(f"\n[STEP 5/5] Saving results...")
    print(f"{'=' * 60}")
    print(f"TRAINING COMPLETED")
    print(f"{'=' * 60}")
    print(f"Total training time: {total_time / 60:.2f} minutes ({total_time:.1f} seconds)")
    print(f"Average time per epoch: {total_time / num_epochs:.1f} seconds")
    print(f"Final test accuracy: {test_acc:.2f}%")
    print(f"Best test accuracy: {best_test_acc:.2f}%")
    print(f"Device used: {device}")
    print(f"{'=' * 60}")

    # # Save model
    # model_path = 'cnn_model.pth'
    # torch.save(model.state_dict(), model_path)
    # print(f"  - Model saved to: {model_path}")

    # Save detailed results
    results_path = 'training_results.txt'
    with open(results_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("CNN Training Results - CIFAR-10\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Device: {device}\n")
        if torch.cuda.is_available():
            f.write(f"GPU Name: {torch.cuda.get_device_name(0)}\n")
            f.write(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\n")
        else:
            f.write(f"CPU Cores: {multiprocessing.cpu_count()}\n")
            if os.environ.get('SLURM_CPUS_PER_TASK'):
                f.write(f"SLURM CPUs: {os.environ.get('SLURM_CPUS_PER_TASK')}\n")

        f.write(f"\nTraining Configuration:\n")
        f.write(f"  - Epochs: {num_epochs}\n")
        f.write(f"  - Batch size: 128\n")
        f.write(f"  - Optimizer: Adam (lr=0.001)\n")
        f.write(f"  - Model parameters: {num_params:,}\n")
        f.write(f"  - DataLoader workers: {num_workers}\n")

        f.write(f"\nResults:\n")
        f.write(f"  - Total training time: {total_time / 60:.2f} minutes ({total_time:.1f} seconds)\n")
        f.write(f"  - Average time per epoch: {total_time / num_epochs:.1f} seconds\n")
        f.write(f"  - Final test accuracy: {test_acc:.2f}%\n")
        f.write(f"  - Best test accuracy: {best_test_acc:.2f}%\n")

        f.write(f"\nDataset:\n")
        f.write(f"  - Training samples: {len(train_dataset)}\n")
        f.write(f"  - Test samples: {len(test_dataset)}\n")
        f.write(f"  - Classes: 10 (CIFAR-10)\n")

    print(f"  - Results saved to: {results_path}")

    print(f"\n{'=' * 60}")
    print("ALL DONE!")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()