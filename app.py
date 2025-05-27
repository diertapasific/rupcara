import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
import os
from gtts import gTTS
import tempfile

# Set correct number of output classes
dataset_dir = "./rupiah_dataset"
class_names = sorted([
    d for d in os.listdir(dataset_dir)
    if d != ".DS_Store" and os.path.isdir(os.path.join(dataset_dir, d))
])
num_classes = len(class_names)  # Pastikan sama dengan model training

# Load model (tanpa pretrained weights, sesuai training)
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load("./r3upiah_model.pth", map_location=torch.device('cpu')))
model.eval()

# Transformasi gambar (tanpa Normalize)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Streamlit UI
st.title("Rupcara - Money Recognition Model")
st.subheader("by Dierta Pasific")
st.write("Upload an image to recognize the denomination.")

img_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if img_file is not None:
    img = Image.open(img_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Transform & Predict
    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)

    predicted_class_index = predicted.item()
    predicted_class_name = class_names[predicted_class_index]
    # Handle Unknown class
    if predicted_class_name.lower() == "unknown":
        st.warning("⚠️ The uploaded image was not recognized as an Indonesian Rupiah banknote. Please upload a clearer or valid image of the currency.")
        tts_message = "Maaf, saya tidak dapat mengenali gambar ini sebagai uang Rupiah. Silahkan coba gunakan gambar lain."
    else:
        nominal_only = predicted_class_name.split('_')[0]
        st.success(f"Prediction Result: Nominal Rp.{nominal_only}")
        tts_message = f"Nominal {nominal_only} Rupiah"

    # Generate Audio
    tts = gTTS(tts_message, lang='id')
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as audio_file:
        tts.save(audio_file.name)
        st.audio(audio_file.name)
