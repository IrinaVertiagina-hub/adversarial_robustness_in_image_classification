import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
import mlflow
import numpy as np
import os
import matplotlib.pyplot as plt

os.environ['MLFLOW_TRACKING_URI'] = 'https://dagshub.com/IrinaVertiagina-hub/adversarial_robustness_in_image_classification.mlflow'
os.environ['MLFLOW_TRACKING_USERNAME'] = 'IrinaVertiagina-hub'
os.environ['MLFLOW_TRACKING_PASSWORD'] = '83a58a5ab0de081c13d2f3197d1b8369b0e0ebb0'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
def load_model(path="models/resnet18_cifar10_best.pth"):
    model = resnet18(weights=None, num_classes=10)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model

# FGSM attack
def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, -2.5, 2.5)
    return perturbed_image

def evaluate_fgsm(model, dataloader, epsilon):
    criterion = nn.CrossEntropyLoss()
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        images = images.clone().detach().requires_grad_(True)

        outputs = model(images)
        loss = criterion(outputs, labels)
        model.zero_grad()
        loss.backward()

        data_grad = images.grad.data
        perturbed = fgsm_attack(images, epsilon, data_grad)

        with torch.no_grad():
            outputs = model(perturbed)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return correct / total

# Main
if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    testset = torchvision.datasets.CIFAR10(root='./data/raw', train=False, download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

    model = load_model()
    epsilons = [0, 0.05, 0.1, 0.2, 0.3]
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    print(f"Clean accuracy: {correct/total:.4f}")

    mlflow.set_experiment("adversarial_robustness")

    with mlflow.start_run(run_name="fgsm_attack"):
        mlflow.log_param("attack", "FGSM")
        mlflow.log_param("baseline_accuracy", 0.8903)

        for eps in epsilons:
            acc = evaluate_fgsm(model, testloader, eps)
            mlflow.log_metric(f"accuracy_eps_{eps}", acc)
            print(f"Epsilon: {eps} | Accuracy: {acc:.4f}")

#visualize

def visualize_attack(model, dataloader, epsilon=0.1, num_images=5):
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']
    
    criterion = nn.CrossEntropyLoss()
    images_shown = 0
    
    fig, axes = plt.subplots(num_images, 2, figsize=(6, num_images * 2.5))
    
    for images, labels in dataloader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        images.requires_grad = True
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        model.zero_grad()
        loss.backward()
        
        perturbed = fgsm_attack(images, epsilon, images.grad.data)
        
        orig_preds = model(images).argmax(1)
        adv_preds = model(perturbed).argmax(1)
        
        for i in range(len(images)):
            if images_shown >= num_images:
                break
            
            orig_img = images[i].detach().cpu().permute(1, 2, 0).numpy()
            adv_img = perturbed[i].detach().cpu().permute(1, 2, 0).numpy()
            
            # Denormalize
            mean = [0.4914, 0.4822, 0.4465]
            std = [0.2023, 0.1994, 0.2010]
            orig_img = orig_img * std + mean
            adv_img = adv_img * std + mean
            orig_img = orig_img.clip(0, 1)
            adv_img = adv_img.clip(0, 1)
            
            axes[images_shown, 0].imshow(orig_img)
            axes[images_shown, 0].set_title(f"Original: {classes[orig_preds[i]]}")
            axes[images_shown, 0].axis('off')
            
            axes[images_shown, 1].imshow(adv_img)
            axes[images_shown, 1].set_title(f"Attacked: {classes[adv_preds[i]]}")
            axes[images_shown, 1].axis('off')
            
            images_shown += 1
        
        if images_shown >= num_images:
            break
    
    plt.tight_layout()
    plt.savefig(f'reports/fgsm_visualization_eps{eps}.png')
    print("Saved to reports/fgsm_visualization.png")


for eps in [0.05, 0.1, 0.2, 0.3]:
    visualize_attack(model, testloader, epsilon=eps, num_images=5)