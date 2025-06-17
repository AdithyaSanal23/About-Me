import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from main import ChangeDetectionModel

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ChangeDetectionModel(num_classes=7).to(device)
model.load_state_dict(torch.load("scd_model.pth", map_location=device))
model.eval()

# Define preprocessing
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])

# Define binary class labels and colors
class_labels = ["Change", "No Change"]
class_colors = ["black", "pink"]
cmap = ListedColormap(class_colors)

# Streamlit UI
st.set_page_config(layout="wide", page_title="Land-Cover Change Detection")

st.title("Change Detection for Land-Cover Classification")
st.write("Upload old and new images to detect land-cover changes.")

col1, col2 = st.columns(2)

with col1:
    old_image = st.file_uploader("Upload Old Image", type=["jpg", "png", "jpeg"])

with col2:
    new_image = st.file_uploader("Upload New Image", type=["jpg", "png", "jpeg"])

if old_image and new_image:
    # Preprocess images
    old_img = Image.open(old_image).convert("RGB")
    new_img = Image.open(new_image).convert("RGB")

    old_tensor = transform(old_img).unsqueeze(0)
    new_tensor = transform(new_img).unsqueeze(0)

    input_tensor = torch.cat([old_tensor, new_tensor], dim=1).to(device)

    # Perform inference
    with torch.no_grad():
        output = model(input_tensor)

    # Post-process output: Convert multi-class to binary (0: No Change, 1: Change)
    change_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()
    binary_mask = np.where(change_mask == 0, 0, 1)  # All non-zero classes become 1 (Change)

    # Visualize results
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

    ax1.imshow(old_img)
    ax1.set_title("Old Image")
    ax1.axis('off')

    ax2.imshow(new_img)
    ax2.set_title("New Image")
    ax2.axis('off')

    im = ax3.imshow(binary_mask, cmap=cmap, vmin=0, vmax=1)
    ax3.set_title("Change Detection Result")
    ax3.axis('off')

    # Add colorbar with binary class labels
    cbar = plt.colorbar(im, ax=ax3, orientation='vertical', aspect=30)
    cbar.set_ticks([0.25, 0.75])  # Midpoints for binary classes
    cbar.set_ticklabels(class_labels)
    cbar.set_label('Classes')

    st.pyplot(fig)

    # Calculate change statistics
    unique, counts = np.unique(binary_mask, return_counts=True)
    total_pixels = binary_mask.size

    st.subheader("Change Statistics")
    no_change_percentage = (counts[np.where(unique == 0)[0][0]] / total_pixels) * 100 if 0 in unique else 0.00
    change_percentage = (counts[np.where(unique == 1)[0][0]] / total_pixels) * 100 if 1 in unique else 0.00

    st.write(f"Change: {no_change_percentage:.2f}%")
    st.write(f"No Change: {change_percentage:.2f}%")
else:
    st.write("Please upload both old and new images to proceed.")

# Add some information about the project
st.sidebar.title("About")
st.sidebar.info(
    "This application uses a deep learning model to detect "
    "land-cover changes between two satellite or aerial images. "
)

# Add instructions
st.sidebar.title("Instructions")
st.sidebar.markdown(
    """
    1. Upload an old image of the area.
    2. Upload a new image of the same area.
    3. View the change detection results and statistics.
    """
)
