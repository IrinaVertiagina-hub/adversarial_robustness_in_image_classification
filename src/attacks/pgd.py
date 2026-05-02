import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
import mlflow
import os

os.environ['MLFLOW_TRACKING_URI'] = 'https://dagshub.com/IrinaVertiagina-hub/adversarial_robustness_in_image_classification.mlflow'
os.environ['MLFLOW_TRACKING_USERNAME'] = 'IrinaVertiagina-hub'
os.environ['MLFLOW_TRACKING_PASSWORD'] = '83a58a5ab0de081c13d2f3197d1b8369b0e0ebb0'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(path="models/resnet18_cifar10_best.pth"):
    model = resnet18(weights=None, num_classes=10)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model

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

def evaluate_pgd(model, dataloader, epsilon, alpha, steps):
    correct = 0
    total = 0
    
    for images, labels in dataloader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        perturbed = pgd_attack(model, images, labels, epsilon, alpha, steps)
        
        with torch.no_grad():
            outputs = model(perturbed)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return correct / total

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    testset = torchvision.datasets.CIFAR10(root='./data/raw', train=False, download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

    model = load_model()

    configs = [
        {"epsilon": 0.05, "alpha": 0.01, "steps": 10},
        {"epsilon": 0.1,  "alpha": 0.01, "steps": 20},
        {"epsilon": 0.2,  "alpha": 0.02, "steps": 20},
    ]

    mlflow.set_experiment("adversarial_robustness")

    with mlflow.start_run(run_name="pgd_attack"):
        mlflow.log_param("attack", "PGD")
        mlflow.log_param("baseline_accuracy", 0.8903)

        for config in configs:
            acc = evaluate_pgd(model, testloader, **config)
            key = f"acc_eps{config['epsilon']}_steps{config['steps']}"
            mlflow.log_metric(key, acc)
            print(f"Epsilon: {config['epsilon']} | Steps: {config['steps']} | Accuracy: {acc:.4f}")