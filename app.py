import streamlit as st
from PIL import Image
import numpy as np
import time
import io
import zipfile

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="CropGuard: Train & Spray", page_icon="🚜", layout="wide")

# --- CSS STYLING ---
st.markdown("""
    <style>
    .main-header {font-size:30px; font-weight:bold; color:#2E8B57;}
    .stButton>button {width: 100%; border-radius: 5px; height: 3em;}
    </style>""", unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---

def simulate_training(epochs, progress_bar, status_text, chart):
    """
    Simulates the training process loop (Forward Pass -> Backward Pass).
    REPLACE THIS with real TensorFlow/PyTorch training loop.
    """
    loss_values = []
    accuracy_values = []
    
    for epoch in range(epochs):
        # Simulate time per epoch
        time.sleep(0.5) 
        
        # specific math to make the graph look like real learning
        loss = 1.0 / (epoch + 1) + np.random.uniform(0, 0.1)
        acc = 1.0 - loss
        
        loss_values.append(loss)
        accuracy_values.append(acc)
        
        # Update UI
        progress_bar.progress((epoch + 1) / epochs)
        status_text.text(f"Epoch {epoch+1}/{epochs} | Loss: {loss:.4f} | Accuracy: {acc:.1%}")
        
        # Update Chart
        chart.line_chart({"Loss": loss_values, "Accuracy": accuracy_values})
    
    return "my_custom_model.h5"

# --- MAIN APP ---

def main():
    # --- SIDEBAR: MODE SELECTION ---
    st.sidebar.title("System Control")
    app_mode = st.sidebar.selectbox("Choose Mode", ["Train New Model", "Run System (Inference)"])
    
    st.sidebar.divider()
    
    # --- MODE 1: TRAIN NEW MODEL ---
    if app_mode == "Train New Model":
        st.title(" Model Training")
        st.markdown("Upload your farm's dataset to create a custom AI for your sprayer.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("1. Upload Dataset")
            st.info("Structure: A .zip file containing folders named 'Crop' and 'Weed'.")
            uploaded_zip = st.file_uploader("Upload Zip File", type="zip")
            
            if uploaded_zip:
                # Basic check of zip content
                with zipfile.ZipFile(uploaded_zip, "r") as z:
                    file_list = z.namelist()
                    img_count = len([f for f in file_list if f.endswith(('.jpg', '.png'))])
                    st.success(f" Dataset Loaded: {img_count} images found.")

        with col2:
            st.subheader("2. Training Config")
            epochs = st.slider("Training Epochs", 1, 50, 10, help="How many times the AI studies the images.")
            batch_size = st.selectbox("Batch Size", [16, 32, 64], index=1)
            learning_rate = st.select_slider("Learning Rate", options=[0.01, 0.001, 0.0001], value=0.001)

        st.divider()
        
        # Start Training Button
        if st.button("Start Training", type="primary", disabled=(not uploaded_zip)):
            st.write("### Training Progress")
            
            # UI Elements for updates
            prog_bar = st.progress(0)
            status_txt = st.empty()
            chart_area = st.empty()
            
            # Run the simulation
            model_name = simulate_training(epochs, prog_bar, status_txt, chart_area)
            
            st.success(" Training Complete!")
            st.balloons()
            
            # Download Button for the trained model
            st.download_button(
                label=" Download Trained Model (.h5)",
                data="Fake Model Bytes",
                file_name="my_farm_model.h5",
                mime="application/octet-stream"
            )

    # --- MODE 2: INFERENCE (THE SPRAYER VIEW) ---
    elif app_mode == "Run System (Inference)":
        st.title("Sprayer Vision System")
        st.markdown("Test the model before deploying to the tractor.")
        
        # Option to upload the model they just trained
        model_file = st.sidebar.file_uploader("Load Model File (.h5)", type=["h5", "pt"])
        
        if not model_file:
            st.warning(" Please upload a trained model file in the sidebar to begin.")
        else:
            st.sidebar.success("Model Active")
        
        # Image Input
        test_image = st.file_uploader("Input Camera Feed (Image)", type=["jpg", "png"])
        
        if test_image:
            col_img, col_data = st.columns([1, 1])
            
            with col_img:
                st.image(test_image, caption="Camera View", use_container_width=True)
            
            with col_data:
                st.subheader("Decision Engine")
                with st.spinner("Processing..."):
                    time.sleep(0.8) # Simulate inference time
                    
                    # Mock Prediction
                    is_weed = np.random.choice([True, False])
                    confidence = np.random.uniform(0.85, 0.99)
                    
                    if is_weed:
                        st.error("DETECTED: WEED")
                        st.metric("Confidence", f"{confidence:.1%}")
                        st.markdown("### 🚿 Action: **SPRAY NOZZLE 4 ON**")
                    else:
                        st.success("DETECTED: CROP")
                        st.metric("Confidence", f"{confidence:.1%}")
                        st.markdown("### Action: **SKIP**")

if __name__ == "__main__":
    main()