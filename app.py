import streamlit as st
from PIL import Image
import numpy as np
import os
import shutil
import yaml
import zipfile
from ultralytics import YOLO
import kagglehub
from kaggle.api.kaggle_api_extended import KaggleApi
import json

# --- KAGGLE CREDENTIALS ---
kaggle_file = "kaggle.json"
if os.path.exists(kaggle_file):
    with open(kaggle_file, "r") as f:
        data = json.load(f)
        os.environ['KAGGLE_USERNAME'] = data['username']
        os.environ['KAGGLE_KEY'] = data['key']
    #current directory
    os.environ['KAGGLE_CONFIG_DIR'] = os.getcwd()
else:
    pass

# --- OUTPUT DIRECTORIES ---
PT_OUTPUT_DIR = os.path.join("exports", "pt")
ONNX_OUTPUT_DIR = os.path.join("exports", "onnx")

# --- HELPER FUNCTIONS ---

def search_kaggle_datasets(query):
    """Search kaggle dataset using api"""
    try:
        api = KaggleApi()
        api.authenticate()
        # Search and return top 10 results sorted by vote 
        datasets = api.dataset_list(search=query, sort_by='votes')
        
        # FIX: Safely extract attributes in case the Kaggle API changes them again
        results = []
        for d in datasets[:10]:
            ref = getattr(d, 'ref', 'Unknown Ref')
            title = getattr(d, 'title', 'Unknown Title')
            size = getattr(d, 'totalBytes', 'Size Unknown') # Use totalBytes or default
            results.append((ref, title, size))
            
        return results
    except Exception as e:
        st.error(f"Kaggle API Error: {e}")
        return []

def setup_dataset_from_kaggle(kaggle_handle):
    """Downloads the dataset from kaggle"""
    data_dir = "temp_dataset"
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    os.makedirs(data_dir)

    try:
        with st.spinner(f"Downloading dataset {kaggle_handle} from Kaggle..."):
            path = kagglehub.dataset_download(kaggle_handle)

            # Copy file from to temp dataset
            for item in os.listdir(path):
                s = os.path.join(path, item)
                d = os.path.join(data_dir, item)
                if os.path.isdir(s):
                    shutil.copytree(s, d)
                else:
                    shutil.copy2(s, d)
            st.success("Dataset downloaded and set up successfully!")
            return True
        
    except Exception as e:
        st.error(f"Failed to download dataset: {e}")
        return False

def setup_dataset_from_zip(uploaded_file):
    """Extracts uploaded zip file to the temp dataset directory."""
    data_dir = "temp_dataset"
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    os.makedirs(data_dir)

    try:
        with st.spinner("Extracting dataset..."):
            with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            return True
    except Exception as e:
        st.error(f"Failed to extract zip: {e}")
        return False

def train_yolo(epochs, batch_size, model_size, progress_bar, status_text, chart):
    """Executes real YOLOv8 training."""
    data_dir = os.path.abspath("temp_dataset")
    
    # 1. Check for existing YAML (Common in Kaggle datasets)
    existing_yaml = None
    for root, dirs, files in os.walk(data_dir):
        if "data.yaml" in files or "dataset.yaml" in files:
            existing_yaml = os.path.join(root, files[files.index("data.yaml") if "data.yaml" in files else files.index("dataset.yaml")])
            break
            
    if existing_yaml:
        yaml_path = existing_yaml
        st.info(f"Using found config: {yaml_path}")
    else:
        # SMART CHECK: Scan the dataset to find where the images are actually hiding
        image_folder = None
        for root, dirs, files in os.walk(data_dir):
            # Check if this folder contains image files
            if any(f.endswith(('.jpg', '.jpeg', '.png')) for f in files):
                image_folder = root
                break
                
        if not image_folder:
            st.error(" Could not find any images in this dataset. It might be corrupt.")
            return None
            
        # Search for 'classes.txt' to get the actual names of the objects
        class_names = {0: 'target'}
        for root, dirs, files in os.walk(data_dir):
            if "classes.txt" in files:
                with open(os.path.join(root, "classes.txt"), "r") as f:
                    classes = [line.strip() for line in f.readlines() if line.strip()]
                    class_names = {i: name for i, name in enumerate(classes)}
                break

        # Generate a custom YAML mapping to the exact folder we found
        relative_img_path = os.path.relpath(image_folder, data_dir)
        # Windows uses backslashes, but YOLO requires forward slashes in the YAML
        relative_img_path = relative_img_path.replace("\\", "/") 
        
        yaml_content = {
            'path': data_dir,
            'train': relative_img_path, 
            'val': relative_img_path, # Use same folder for validation if no split exists
            'names': class_names
        }
        yaml_path = os.path.join(data_dir, "generated_data.yaml")
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_content, f)
            
        st.info(f"Auto-generated config mapping to folder: {relative_img_path}")

    # 2. Initialize Model 
    model_map = {"Nano (n)": "yolov8n.pt", "Small (s)": "yolov8s.pt", "Medium (m)": "yolov8m.pt"}
    model_name = model_map.get(model_size, "yolov8n.pt")
    model = YOLO(model_name)

    # 3. Define Callback
    metrics_history = {"box_loss": [], "mAP50": []}

    def on_train_epoch_end(trainer):
        "Called at the end of each epoch to update progress and metrics."
        current_epoch = trainer.epoch + 1
        total_epochs = trainer.epochs 

        # Accessing internal trainer metrics
        loss = trainer.loss_items[0].item() if hasattr(trainer, 'loss_items') else 0
        map50 = trainer.metrics.get("metrics/mAP50(B)", 0)

        metrics_history["box_loss"].append(loss)
        metrics_history["mAP50"].append(map50)

        # Update Streamlit Elements 
        progress = current_epoch / total_epochs
        progress_bar.progress(progress)
        status_text.text(f"Epoch {current_epoch}/{total_epochs} | Box Loss: {loss:.4f}, mAP50: {map50:.1%}")    
        chart.line_chart(metrics_history)

    # Register the callback 
    model.clear_callback("on_train_epoch_end")  
    model.add_callback("on_train_epoch_end", on_train_epoch_end)

    # 5. Start Training
    try:
        results = model.train(
            data = yaml_path,
            epochs = epochs,
            batch = batch_size,
            imgsz = 640,
            exist_ok = True,
            verbose = False
        )

        # Return path to the best saved model weights
        return os.path.join(results.save_dir, "weights", "best.pt")
    
    except Exception as e:
        st.error(f" Training failed: {e}")
        return None

# --- SETUP: CLEANUP PREVIOUS RUNS ---
if os.path.exists("runs"): 
    shutil.rmtree('runs')

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="CropGuard: Train & Spray", page_icon="g", layout="wide")

# --- CSS STYLING ---
st.markdown("""
    <style>
    .main-header {font-size:30px; font-weight:bold; color:#2E8B57;}
    .stButton>button {width: 100%; border-radius: 5px; height: 3em;}
    </style>""", unsafe_allow_html=True)

# --- MAIN APP ---

def main():
    st.sidebar.title("System Control")
    app_mode = st.sidebar.selectbox("Choose Mode", ["Train New Model", "Run System (Inference)"])
    st.sidebar.divider()
    
    # Initialize session state for dataset readiness
    if "dataset_ready" not in st.session_state:
        st.session_state.dataset_ready = False
    
    if app_mode == "Train New Model":
        st.title(" Model Training Workshop")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("1. Data Source")
            source_type = st.radio("Select Source", ["Search Kaggle", "Upload Zip", "Draw Custom Dataset"])
            
            if source_type == "Upload Zip":
                uploaded_zip = st.file_uploader("Upload Dataset (.zip)", type="zip")
                # Added an extraction button so it doesn't extract repeatedly on reruns
                if uploaded_zip and st.button("Extract Dataset"):
                    if setup_dataset_from_zip(uploaded_zip):
                        st.success("Zip extracted successfully.")
                        st.session_state.dataset_ready = True 
                        
            elif source_type == "Search Kaggle":
                st.markdown("Find datasets directly from Kaggle.")
                search_query = st.text_input("Search (e.g., 'fruit detection', 'potholes')", value="fruit quality")
                
                if search_query:
                    results = search_kaggle_datasets(search_query)
                    
                    if results:
                        options = {f"{r[1]} ({r[2]})": r[0] for r in results}
                        selected_label = st.selectbox("Select a Dataset", options.keys())
                        selected_handle = options[selected_label]
                        st.caption(f"Handle: `{selected_handle}`")
                        
                        if st.button("Download & Prepare"):
                            if setup_dataset_from_kaggle(selected_handle):
                                st.session_state.dataset_ready = True
                    else:
                        st.warning("No results found. (Check your kaggle.json key)")
            
            elif source_type == "Draw Custom Dataset":
                st.markdown("Use the built-in desktop tool to annotate your own images.")
                
                # Button to launch your PySide6 app!
                if st.button(" Launch Annotation Tool"):
                    import subprocess
                    import sys
                    subprocess.Popen([sys.executable, "bbox.py"])
                    st.toast("Opening Bounding Box Tool...")
                    
                st.divider()
                st.markdown("After generating your `dataset.yaml` in the tool, paste the path to that folder below:")
                
                custom_folder = st.text_input("Dataset Folder Path:")
                
                if custom_folder and st.button("Load Custom Dataset"):
                    # Check if the folder exists and contains the yaml
                    if os.path.exists(custom_folder) and any(f.endswith('.yaml') for f in os.listdir(custom_folder)):
                        # Copy contents to Streamlit's workspace
                        data_dir = "temp_dataset"
                        if os.path.exists(data_dir):
                            shutil.rmtree(data_dir)
                        shutil.copytree(custom_folder, data_dir)
                        
                        st.success("Custom dataset loaded successfully!")
                        st.session_state.dataset_ready = True
                    else:
                        st.error(" Invalid folder path or missing .yaml file.")

        with col2:
            st.subheader("2. Hyperparameters")
            epochs = st.slider("Training Epochs", 1, 50, 5)
            batch_size = st.selectbox("Batch Size", [16, 32, 64], index=0)
            model_size = st.select_slider("Model Architecture", options=["Nano (n)", "Small (s)", "Medium (m)"], value="Nano (n)")

        st.divider()
        
        # Use session state to check if the button should be enabled
        if st.button("Start Training", type="primary", disabled=not st.session_state.dataset_ready):
            st.write("### Training Progress")
            prog_bar = st.progress(0)
            status_txt = st.empty()
            chart_area = st.empty()
            
            best_model_path = train_yolo(epochs, batch_size, model_size, prog_bar, status_txt, chart_area)
            
            if best_model_path:
                st.success("Training Complete!")
                st.balloons()
                pt_path = best_model_path
                try:
                    os.makedirs(PT_OUTPUT_DIR, exist_ok=True)
                    pt_path = os.path.join(PT_OUTPUT_DIR, "custom_model.pt")
                    shutil.copy2(best_model_path, pt_path)
                    st.success(f"PT copy saved to: {pt_path}")
                except Exception as e:
                    st.warning(f"PT copy skipped: {e}")
                onnx_path = None
                try:
                    os.makedirs(ONNX_OUTPUT_DIR, exist_ok=True)
                    trained_model = YOLO(best_model_path)
                    exported_path = trained_model.export(format="onnx")
                    onnx_path = os.path.join(ONNX_OUTPUT_DIR, "custom_model.onnx")
                    shutil.copy2(exported_path, onnx_path)
                    st.success(f"ONNX export saved to: {onnx_path}")
                except Exception as e:
                    st.warning(f"ONNX export skipped: {e}")
                with open(pt_path, "rb") as f:
                    st.download_button(
                        label="⬇ Download Model (.pt)",
                        data=f.read(),
                        file_name="custom_model.pt",
                        mime="application/octet-stream"
                    )
                if onnx_path and os.path.exists(onnx_path):
                    with open(onnx_path, "rb") as f:
                        st.download_button(
                            label="Download Model (.onnx)",
                            data=f.read(),
                            file_name="custom_model.onnx",
                            mime="application/octet-stream"
                        )

    elif app_mode == "Run System (Inference)":
        st.title(" Inference Mode")
        st.info("Switch to 'Train New Model' to build your custom detector.")

if __name__ == "__main__":
    main()
