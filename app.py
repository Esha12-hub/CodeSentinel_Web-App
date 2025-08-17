<<<<<<< HEAD
import torch
from torchvision import models, transforms
from PIL import Image
import streamlit as st
import torch.nn as nn

model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 5)

state_dict = torch.load(
    r"C:\Users\Public\Documents\VS_Projects\Web_App\model_state_dict.pth",
    map_location=torch.device('cpu'),
    weights_only=True  # Safe in PyTorch 2.6+
)
model.load_state_dict(state_dict)
model.eval()

# 3️⃣ Class labels
class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

# 4️⃣ Transform for input images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 5️⃣ Streamlit UI
st.set_page_config(page_title="🌸 Flower Classifier", layout="centered")
st.title("🌼 AI Flower Identifier")
st.write("Upload a flower image to get its name using ResNet18!")

uploaded_file = st.file_uploader("📤 Upload a flower image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=False)

    if st.button("Predict"):
        with st.spinner("Predicting..."):
            img = transform(image).unsqueeze(0)
            with torch.no_grad():
                outputs = model(img)
                _, predicted = torch.max(outputs, 1)
                label = class_names[predicted.item()]
            st.success(f"✅ Predicted Flower: **{label.upper()}**")
=======
import torch
from torchvision import models, transforms
from PIL import Image
import streamlit as st
import torch.nn as nn

model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 5)

state_dict = torch.load(
    r"C:\Users\Public\Documents\VS_Projects\Web_App\model_state_dict.pth",
    map_location=torch.device('cpu'),
    weights_only=True  # Safe in PyTorch 2.6+
)
model.load_state_dict(state_dict)
model.eval()

# 3️⃣ Class labels
class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

# 4️⃣ Transform for input images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 5️⃣ Streamlit UI
st.set_page_config(page_title="🌸 Flower Classifier", layout="centered")
st.title("🌼 AI Flower Identifier")
st.write("Upload a flower image to get its name using ResNet18!")

uploaded_file = st.file_uploader("📤 Upload a flower image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=False)

    if st.button("Predict"):
        with st.spinner("Predicting..."):
            img = transform(image).unsqueeze(0)
            with torch.no_grad():
                outputs = model(img)
                _, predicted = torch.max(outputs, 1)
                label = class_names[predicted.item()]
            st.success(f"✅ Predicted Flower: **{label.upper()}**")
>>>>>>> 276e653 (Initial commit)
