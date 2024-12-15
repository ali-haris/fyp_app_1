import streamlit as st
from PIL import Image
from ultralytics import YOLO  # Replace with the YOLO version you're using

# Caching the model loading to optimize performance
@st.cache_resource
def load_model():
    # Replace "path/to/your/trained/model.pt" with your actual model path
    return YOLO("./best.pt")

# Load the YOLO model
model = load_model()

# Title and description
st.title("Outdoor Object Detection for Visually Impaired People")
st.write("Upload an image for object detection.")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    if uploaded_file.type.startswith("image"):
        # Process image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        # Perform inference
        results = model(image)
        # Display detection results
        st.image(results[0].plot(), caption="Detected Objects")
    elif uploaded_file.type == "video/mp4":
            st.write('Video is not supported for now')
        # with open("temp_video.mp4", "wb") as f:
        #     f.write(uploaded_file.read())
        # # Perform inference on video
        # results = model("temp_video.mp4", save=True)
        # # Display video with detections
        # st.video("runs/detect/pred/temp_video.mp4")
