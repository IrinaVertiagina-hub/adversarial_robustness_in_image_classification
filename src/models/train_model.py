import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
import mlflow
import mlflow.pytorch

# Config
CONFIGS = [
    {"batch_size": 64,  "lr": 0.1, "patience": 10},
]

EPOCHS = 200
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data with augmentation
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

testset = torchvision.datasets.CIFAR10(root='./data/raw', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

def train(config):
    trainset = torchvision.datasets.CIFAR10(root='./data/raw', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config["batch_size"], shuffle=True, num_workers=0)

    model = resnet18(weights=None, num_classes=10).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_acc = 0.0
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        scheduler.step()

        # Eval
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        acc = correct / total
        print(f"Epoch {epoch+1} | Acc: {acc:.4f} | Best: {best_acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            patience_counter = 0
            torch.save(model.state_dict(), "models/best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                print(f"Early stopping at epoch {epoch+1}")
                break

    return best_acc, model

# Run experiments
mlflow.set_experiment("adversarial_robustness")

for config in CONFIGS:
    with mlflow.start_run(run_name=f"bs{config['batch_size']}_lr{config['lr']}_augmented"):
        mlflow.log_params(config)
        best_acc, model = train(config)
        mlflow.log_metric("best_accuracy", best_acc)
        mlflow.pytorch.log_model(model, "model")
        print(f"Config {config} → Best Acc: {best_acc:.4f}")

print("All runs done. Run 'py -m mlflow ui' to compare.")