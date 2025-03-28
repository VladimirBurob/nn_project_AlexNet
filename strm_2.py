import streamlit as st
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import time
import urllib.request
import requests
from io import BytesIO
def app():
    @st.cache_data
    def load_imagenet_labels():
        url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
        with urllib.request.urlopen(url) as f:
            labels = [line.decode("utf-8").strip() for line in f]
        return labels


    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
    ])

    @st.cache_resource
    def load_model(name):
        if name == "resnet18":
            model = models.resnet18(pretrained=True)
        elif name == "densenet121":
            model = models.densenet121(pretrained=True)
        else:
            raise ValueError("Unknown model")
        model.eval()
        return model

    resnet = load_model("resnet18")
    densenet = load_model("densenet121")
    labels = load_imagenet_labels()

    def predict(model, image):
        image = transform(image).unsqueeze(0)
        start = time.time()
        with torch.no_grad():
            outputs = model(image)
            probs = torch.nn.functional.softmax(outputs[0], dim=0)
            top5 = torch.topk(probs, 5)
        end = time.time()
        elapsed = end - start
        return [(labels[i], float(probs[i])) for i in top5.indices], elapsed

    st.title("🧠 Классификация изображений двумя моделями")

    st.subheader("🔗 Загрузка изображения по ссылке")
    image_url = st.text_input("Вставьте ссылку на изображение")
    images = []

    if image_url:
        try:
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content)).convert("RGB")
            images.append(("Изображение по ссылке", image))
        except:
            st.error("❌ Не удалось загрузить изображение по ссылке.")

    st.subheader("🖼 Загрузка изображений с компьютера")
    uploaded_files = st.file_uploader("Загрузите одно или несколько изображений", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    for file in uploaded_files:
        image = Image.open(file).convert("RGB")
        images.append((file.name, image))

    if images:
        for name, image in images:
            st.divider()
            st.image(image, caption=name, use_container_width=True)

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📦 ResNet18:")
                top5_rn, time_rn = predict(resnet, image)
                for label, prob in top5_rn:
                    st.write(f"**{label}** — {prob:.4f}")
                st.success(f"⏱ Время инференса: {time_rn:.3f} сек")

            with col2:
                st.subheader("📦 DenseNet121:")
                top5_dn, time_dn = predict(densenet, image)
                for label, prob in top5_dn:
                    st.write(f"**{label}** — {prob:.4f}")
                st.success(f"⏱ Время инференса: {time_dn:.3f} сек")
