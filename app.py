import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
import os
from gtts import gTTS
import tempfile
import os

# Set the correct number of output classes (22 classes in your case)
num_classes = 22

# Load class names from the dataset directory
dataset_dir = "/rupiah_dataset"
class_names = sorted(os.listdir(dataset_dir))  # Get class names from folder structure

# Load the pretrained ResNet model
model = models.resnet18(weights='IMAGENET1K_V1')  # Load a pretrained ResNet18 model
model.fc = nn.Linear(model.fc.in_features, num_classes)  # Adjust the final layer to match the number of classes

# Load the model weights from the checkpoint
model.load_state_dict(torch.load("./r3upiah_model.pth"))
model.eval()

# Define image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Streamlit interface
st.title("Rupcara - Money Recognition Model")

st.write("Upload an image to recognize the denomination.")

img_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if img_file is not None:
    img = Image.open(img_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Transform and predict
    img_tensor = transform(img).unsqueeze(0)
    output = model(img_tensor)
    _, predicted = torch.max(output, 1)

    # Get the predicted class label from class_names
    predicted_class_index = predicted.item()
    predicted_class_name = class_names[predicted_class_index]
    nominal_only = predicted_class_name.split('_')[0]

    # Display the predicted denomination
    st.write(f"Prediction Result: Nominal Rp.{nominal_only}")

    # Create the audio output
    tts = gTTS(f"Nominal {nominal_only} Rupiah", lang='id')
    
    # Save the audio to a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as audio_file:
        tts.save(audio_file.name)
        st.audio(audio_file.name)  # Play the audio using Streamlit
