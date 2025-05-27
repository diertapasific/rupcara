import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
import os
from gtts import gTTS
import tempfile

# Set the correct number of output classes (including 'Unknown')
num_classes = 23  # Make sure this matches your trained model

# Load class names from the dataset directory
dataset_dir = "./rupiah_dataset"
class_names = sorted(os.listdir(dataset_dir))  # Example: ["1000_2016", "5000_2020", "Unknown"]

# In app.py
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load("./r3upiah_model_v2.pth"))
model.eval()

# Define image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Streamlit interface
st.title("Rupcara - Money Recognition Model")
st.subheader("by Dierta Pasific")
st.write("Upload an image to recognize the denomination.")

img_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if img_file is not None:
    img = Image.open(img_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Transform and predict
    img_tensor = transform(img).unsqueeze(0)
    output = model(img_tensor)
    _, predicted = torch.max(output, 1)

    predicted_class_index = predicted.item()
    predicted_class_name = class_names[predicted_class_index]

    # Handle Unknown class
    if predicted_class_name.lower() == "unknown":
        st.warning("⚠️ The uploaded image was not recognized as an Indonesian Rupiah banknote. Please upload a clearer or valid image of the currency.")
        tts_message = "Maaf, saya tidak dapat mengenali gambar ini sebagai uang Rupiah. Silakan coba gambar lain."
    else:
        nominal_only = predicted_class_name.split('_')[0]
        tahun_only = predicted_class_name.split('_')[1] if '_' in predicted_class_name else ''
        st.success(f"Prediction Result: Nominal Rp.{nominal_only} (Tahun Emisi: {tahun_only})")
        tts_message = f"Nominal {nominal_only} Rupiah, tahun emisi {tahun_only}"

    # Generate audio
    tts = gTTS(tts_message, lang='id')
    with tempfile.NamedTemporaryFile(delete=False) as audio_file:
        tts.save(audio_file.name)
        st.audio(audio_file.name)