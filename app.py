import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18
from PIL import Image
import numpy as np

DEVICE = torch.device("cpu")
CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']


@st.cache_resource
def load_models():
    baseline = resnet18(weights=None, num_classes=10)
    baseline.load_state_dict(torch.load("models/resnet18_cifar10_best.pth", map_location=DEVICE))
    baseline.eval()

    defended = resnet18(weights=None, num_classes=10)
    defended.load_state_dict(torch.load("models/adversarial_trained.pth", map_location=DEVICE))
    defended.eval()

    return baseline, defended


def preprocess(image):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    return transform(image).unsqueeze(0)


def fgsm_attack(image, epsilon, data_grad):
    return image + epsilon * data_grad.sign()


def predict(model, tensor):
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)
        pred = probs.argmax(1).item()
    return CLASSES[pred], probs[0][pred].item()


def predict_with_attack(model, tensor, epsilon):
    tensor = tensor.clone().detach().requires_grad_(True)
    output = model(tensor)
    loss = nn.CrossEntropyLoss()(output, output.argmax(1))
    model.zero_grad()
    loss.backward()
    adv_tensor = fgsm_attack(tensor, epsilon, tensor.grad.data)

    with torch.no_grad():
        output = model(adv_tensor)
        probs = torch.softmax(output, dim=1)
        pred = probs.argmax(1).item()

    return CLASSES[pred], probs[0][pred].item(), adv_tensor


def tensor_to_image(tensor):
    img = tensor.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2023, 0.1994, 0.2010])
    img = img * std + mean
    return img.clip(0, 1)


# UI
st.title("Adversarial Robustness Demo")
st.markdown("Upload an image and see how adversarial attacks affect model predictions.")

baseline_model, defended_model = load_models()

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
epsilon = st.slider("Epsilon (attack strength)", 0.0, 0.3, 0.1, 0.01)

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    tensor = preprocess(image)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Original")
        st.image(image, use_container_width=True)
        pred, conf = predict(baseline_model, tensor)
        st.markdown(f"**Prediction:** {pred}")
        st.markdown(f"**Confidence:** {conf:.2%}")

    with col2:
        st.subheader("After FGSM Attack")
        adv_pred, adv_conf, adv_tensor = predict_with_attack(baseline_model, tensor, epsilon)
        adv_img = tensor_to_image(adv_tensor)
        st.image(image, use_container_width=True)
        st.markdown(f"**Prediction:** {adv_pred}")
        st.markdown(f"**Confidence:** {adv_conf:.2%}")

    with col3:
        st.subheader("Defended Model")
        def_pred, def_conf, _ = predict_with_attack(defended_model, tensor, epsilon)
        st.image(image, use_container_width=True)
        st.markdown(f"**Prediction:** {def_pred}")
        st.markdown(f"**Confidence:** {def_conf:.2%}")

    if adv_pred != pred:
        st.error(f"Attack successful! {pred} → {adv_pred}")
    else:
        st.success("Attack failed — model prediction unchanged")