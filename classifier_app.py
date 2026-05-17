import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'classification'))
from train import get_model

MODEL_PATH = "models/stone_classifier.pth"
CLASSES = ["Non-Stone", "Stone"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    model = get_model(len(CLASSES))
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

def predict(image, model):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
    return CLASSES[predicted.item()], confidence.item()

st.title("Kidney Stone Detection 🩺")
st.write("Upload a medical image (like an ultrasound or CT scan) to detect if a kidney stone is present.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg", "bmp", "tif", "tiff"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    st.write("Analyzing...")
    
    try:
        model = load_model()
        prediction, confidence = predict(image, model)

        if prediction == "Stone":
            st.error(f"Prediction: **{prediction}**")
        else:
            st.success(f"Prediction: **{prediction}**")
            
        st.write(f"Confidence: {confidence*100:.2f}%")
        
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        st.info("Make sure you have trained your model first and it is saved at 'models/stone_classifier.pth'.")
