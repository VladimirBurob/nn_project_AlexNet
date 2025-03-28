import streamlit as st
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import time

def app():
    classes = [
        'Annual Crop', 'Forest', 'Herbaceous Vegetation', 'Highway', 'Industrial',
        'Pasture', 'Permanent Crop', 'Residential', 'River', 'Sea/Lake'
    ]

    device = torch.device("mps")

    @st.cache_resource
    def load_model():
        model = models.resnet18(pretrained=True)
        model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
        model.load_state_dict(torch.load("/Users/vladimir/Desktop/ds_bootcamp/nn_project_AlexNet/models/model_eurosat.pt", map_location=device))
        model.to(device)
        model.eval()
        return model

    model = load_model()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    def predict(image: Image.Image):
        image = transform(image).unsqueeze(0).to(device)
        start = time.time()
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
        elapsed = time.time() - start
        return classes[predicted.item()], elapsed

    st.title("🛰 Классификация спутниковых изображений")
    st.write("Загрузите изображение со спутника, и модель предскажет 1 наиболее вероятный класс EuroSAT.")

    uploaded_file = st.file_uploader("Загрузите изображение", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Загруженное изображение", use_container_width=True)

        st.subheader("Результат:")
        label, elapsed = predict(image)
        st.write(f"🧠 **Предсказанный класс:** {label}")
        st.caption(f"⏱ Время инференса: {elapsed:.3f} сек")