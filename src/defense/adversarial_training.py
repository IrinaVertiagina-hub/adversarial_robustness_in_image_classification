import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
import mlflow
import mlflow.pytorch
import os

from kaggle_secrets import UserSecretsClient
secrets = UserSecretsClient()
token = secrets.get_secret("DAGSHUB_TOKEN")



os.environ['MLFLOW_TRACKING_URI'] = 'https://dagshub.com/IrinaVertiagina-hub/adversarial_robustness_in_image_classification.mlflow'
os.environ['MLFLOW_TRACKING_USERNAME'] = 'IrinaVertiagina-hub'
os.environ['MLFLOW_TRACKING_PASSWORD'] = token

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPSILON = 0.1
ALPHA = 0.01
PGD_STEPS = 7

def pgd_attack(model, images, labels, epsilon, alpha, steps):
    criterion = nn.CrossEntropyLoss()
    x_adv = images.clone().detach() + torch.empty_like(images).uniform_(-epsilon, epsilon)
    x_adv = torch.clamp(x_adv, images - epsilon, images + epsilon)

    for _ in range(steps):
        x_adv = x_adv.clone().detach().requires_grad_(True)
        outputs = model(x_adv)
        loss = criterion(outputs, labels)
        model.zero_grad()
        loss.backward()
        x_adv = x_adv + alpha * x_adv.grad.sign()
        x_adv = torch.clamp(x_adv, images - epsilon, images + epsilon)
        x_adv = x_adv.detach()

    return x_adv

def evaluate(model, dataloader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return correct / total

def evaluate_adversarial(model, dataloader, epsilon, alpha, steps):
    model.eval()
    correct = total = 0
    for images, labels in dataloader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        perturbed = pgd_attack(model, images, labels, epsilon, alpha, steps)
        with torch.no_grad():
            outputs = model(perturbed)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return correct / total


os.makedirs("/kaggle/working/models", exist_ok=True)
if __name__ == "__main__":
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    trainset = torchvision.datasets.CIFAR10(root='/kaggle/input/datasets/pankrzysiu/cifar10-python/', train=True, download=False, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='/kaggle/input/datasets/pankrzysiu/cifar10-python/', train=False, download=False, transform=transform_test)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=0)

    model = resnet18(weights=None, num_classes=10).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    best_acc = 0.0
    patience = 10
    patience_counter = 0

    mlflow.set_experiment("adversarial_robustness")

    with mlflow.start_run(run_name="adversarial_training"):
        mlflow.log_params({
            "epsilon": EPSILON,
            "alpha": ALPHA,
            "pgd_steps": PGD_STEPS,
            "optimizer": "SGD",
            "scheduler": "CosineAnnealing"
        })

        for epoch in range(50):
            model.train()
            for images, labels in trainloader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                adv_images = pgd_attack(model, images, labels, EPSILON, ALPHA, PGD_STEPS)
                model.train()
                optimizer.zero_grad()
                outputs = model(adv_images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            scheduler.step()

            clean_acc = evaluate(model, testloader)
            print(f"Epoch {epoch+1}/50 | Clean Acc: {clean_acc:.4f}")

            if clean_acc > best_acc:
                best_acc = clean_acc
                patience_counter = 0
                torch.save(model.state_dict(), "models/adversarial_trained.pth")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        adv_acc = evaluate_adversarial(model, testloader, EPSILON, ALPHA, PGD_STEPS)
        mlflow.log_metric("best_clean_accuracy", best_acc)
        mlflow.log_metric("adversarial_accuracy", adv_acc)
        mlflow.pytorch.log_model(model, "model")
        print(f"Best clean acc: {best_acc:.4f} | Adversarial acc: {adv_acc:.4f}")

torch.save(model.state_dict(), "/kaggle/working/models/adversarial_trained.pth")