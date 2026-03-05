import unittest
from unittest.mock import MagicMock, patch
import os
import shutil

# Mock heavy dependencies BEFORE importing app
import sys
sys.modules['ultralytics'] = MagicMock()
sys.modules['kagglehub'] = MagicMock()
sys.modules['kaggle'] = MagicMock()
sys.modules['kaggle.api.kaggle_api_extended'] = MagicMock()

# Now import the functions to test
from app import setup_dataset_from_zip, search_kaggle_datasets, setup_dataset_from_kaggle, train_yolo

class TestCropGuardApp(unittest.TestCase):
    
    def setUp(self):
        self.test_data_dir = "temp_dataset"
        if os.path.exists(self.test_data_dir):
            shutil.rmtree(self.test_data_dir)
            
    def tearDown(self):
        if os.path.exists(self.test_data_dir):
            shutil.rmtree(self.test_data_dir)

    def test_directory_creation(self):
        """Verify that the temp directory is correctly initialized."""
        os.makedirs(self.test_data_dir)
        self.assertTrue(os.path.exists(self.test_data_dir))

    @patch('zipfile.ZipFile')
    @patch('app.st') # Mock Streamlit to prevent UI errors during headless testing
    def test_setup_dataset_from_zip(self, mock_st, mock_zip):
        """Test zip extraction logic without needing a real zip file."""
        mock_file = MagicMock()
        result = setup_dataset_from_zip(mock_file)
        self.assertTrue(result)
        self.assertTrue(os.path.exists(self.test_data_dir))

    @patch('app.KaggleApi')
    @patch('app.st')
    def test_search_kaggle_datasets_success(self, mock_st, mock_kaggle_api_class):
        """Test successful search of Kaggle datasets and data parsing."""
        # Setup mock Kaggle API instance
        mock_api_instance = MagicMock()
        mock_kaggle_api_class.return_value = mock_api_instance
        
        # Setup mock dataset items returned by dataset_list()
        mock_dataset = MagicMock()
        mock_dataset.ref = 'user/fruit-quality'
        mock_dataset.title = 'Fruit Quality Dataset'
        mock_dataset.totalBytes = '150MB'
        
        mock_api_instance.dataset_list.return_value = [mock_dataset]
        
        # Execute function
        results = search_kaggle_datasets('fruit quality')
        
        # Assertions
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0], ('user/fruit-quality', 'Fruit Quality Dataset', '150MB'))
        mock_api_instance.authenticate.assert_called_once()
        mock_api_instance.dataset_list.assert_called_once_with(search='fruit quality', sort_by='votes')

    @patch('app.KaggleApi')
    @patch('app.st')
    def test_search_kaggle_datasets_exception(self, mock_st, mock_kaggle_api_class):
        """Test Kaggle search graceful failure handling."""
        mock_api_instance = MagicMock()
        mock_kaggle_api_class.return_value = mock_api_instance
        mock_api_instance.authenticate.side_effect = Exception("API Auth Error")
        
        results = search_kaggle_datasets('fruit quality')
        
        # Should return an empty list and call Streamlit's error function
        self.assertEqual(results, [])
        mock_st.error.assert_called_once()

    @patch('app.kagglehub')
    @patch('app.st')
    def test_setup_dataset_from_kaggle_success(self, mock_st, mock_kagglehub):
        """Test downloading and setting up a dataset from Kaggle."""
        # Create a mock download path with dummy files to simulate a real download
        mock_download_path = "mock_kaggle_download_dir"
        os.makedirs(mock_download_path, exist_ok=True)
        with open(os.path.join(mock_download_path, "dummy_data.txt"), "w") as f:
            f.write("test data")
            
        mock_kagglehub.dataset_download.return_value = mock_download_path
        
        result = setup_dataset_from_kaggle('user/fruit-quality')
        
        self.assertTrue(result)
        # Verify the file was successfully copied to our temp_dataset folder
        self.assertTrue(os.path.exists(os.path.join(self.test_data_dir, "dummy_data.txt")))
        
        # Cleanup mock download path
        shutil.rmtree(mock_download_path)

    @patch('app.kagglehub')
    @patch('app.st')
    def test_setup_dataset_from_kaggle_exception(self, mock_st, mock_kagglehub):
        """Test Kaggle download failure handling."""
        mock_kagglehub.dataset_download.side_effect = Exception("Download Error")
        
        result = setup_dataset_from_kaggle('user/fruit-quality')
        
        self.assertFalse(result)
        mock_st.error.assert_called_once()

    @patch('app.YOLO')
    @patch('app.st')
    def test_train_yolo_no_images(self, mock_st, mock_yolo_class):
        """Test train_yolo when no images are found in the dataset (corrupt data)."""
        # Create an empty temp_dataset
        os.makedirs(self.test_data_dir, exist_ok=True)
        
        # Execute
        result = train_yolo(5, 16, "Nano (n)", MagicMock(), MagicMock(), MagicMock())
        
        # Assertions
        self.assertIsNone(result)
        mock_st.error.assert_called_once()

    @patch('app.YOLO')
    @patch('app.st')
    def test_train_yolo_existing_yaml(self, mock_st, mock_yolo_class):
        """Test train_yolo correctly identifies and uses an existing data.yaml."""
        os.makedirs(self.test_data_dir, exist_ok=True)
        yaml_path = os.path.join(self.test_data_dir, "data.yaml")
        with open(yaml_path, "w") as f:
            f.write("mock yaml content")
            
        # Mock YOLO model and its train method
        mock_model = MagicMock()
        mock_yolo_class.return_value = mock_model
        mock_results = MagicMock()
        mock_results.save_dir = "mock_save_dir"
        mock_model.train.return_value = mock_results
        
        # Execute
        result = train_yolo(5, 16, "Nano (n)", MagicMock(), MagicMock(), MagicMock())
        
        # Assertions
        self.assertEqual(result, os.path.join("mock_save_dir", "weights", "best.pt"))
        mock_model.train.assert_called_once()
        # Verify it used the correct YAML
        mock_st.info.assert_called_with(f"Using found config: {os.path.abspath(yaml_path)}")

    @patch('app.YOLO')
    @patch('app.st')
    def test_train_yolo_auto_generate_yaml_and_callback(self, mock_st, mock_yolo_class):
        """Test train_yolo generating a yaml when images are found but no config exists, and test the callback."""
        # Create a mock dataset with images and a classes.txt
        img_dir = os.path.join(self.test_data_dir, "images")
        os.makedirs(img_dir, exist_ok=True)
        with open(os.path.join(img_dir, "test1.jpg"), "w") as f:
            f.write("fake image")
        with open(os.path.join(self.test_data_dir, "classes.txt"), "w") as f:
            f.write("weed\ncrop\n")
            
        # Mocks
        mock_model = MagicMock()
        mock_yolo_class.return_value = mock_model
        mock_results = MagicMock()
        mock_results.save_dir = "mock_save_dir"
        mock_model.train.return_value = mock_results
        
        mock_progress_bar = MagicMock()
        mock_status_text = MagicMock()
        mock_chart = MagicMock()
        
        # Execute
        result = train_yolo(5, 16, "Nano (n)", mock_progress_bar, mock_status_text, mock_chart)
        
        # 1. Assertions for YAML generation
        self.assertEqual(result, os.path.join("mock_save_dir", "weights", "best.pt"))
        generated_yaml = os.path.join(self.test_data_dir, "generated_data.yaml")
        self.assertTrue(os.path.exists(generated_yaml))
        
        # 2. Test the nested callback function manually to ensure UI updates are covered
        args, kwargs = mock_model.add_callback.call_args
        callback_name = args[0]
        callback_fn = args[1]
        
        self.assertEqual(callback_name, "on_train_epoch_end")
        
        # Simulate a trainer object passed to the callback at the end of epoch 1
        mock_trainer = MagicMock()
        mock_trainer.epoch = 0 # 0-indexed, so 0 is Epoch 1
        mock_trainer.epochs = 5
        
        # Mocking loss item tensor
        mock_loss_tensor = MagicMock()
        mock_loss_tensor.item.return_value = 1.25
        mock_trainer.loss_items = [mock_loss_tensor]
        mock_trainer.metrics = {"metrics/mAP50(B)": 0.85}
        
        # Trigger the nested callback
        callback_fn(mock_trainer)
        
        # Verify the Streamlit UI elements were updated by the callback
        mock_progress_bar.progress.assert_called_with(0.2) # 1 / 5 epochs
        mock_status_text.text.assert_called_with("Epoch 1/5 | Box Loss: 1.2500, mAP50: 85.0%")
        mock_chart.line_chart.assert_called_once()

    @patch('app.YOLO')
    @patch('app.st')
    def test_train_yolo_training_exception(self, mock_st, mock_yolo_class):
        """Test train_yolo handling an unexpected crash during training."""
        os.makedirs(self.test_data_dir, exist_ok=True)
        with open(os.path.join(self.test_data_dir, "data.yaml"), "w") as f:
            f.write("mock yaml content")
            
        # Force the train method to throw an error (e.g. Out of Memory)
        mock_model = MagicMock()
        mock_yolo_class.return_value = mock_model
        mock_model.train.side_effect = Exception("CUDA Out of Memory")
        
        # Execute
        result = train_yolo(5, 16, "Nano (n)", MagicMock(), MagicMock(), MagicMock())
        
        # Assertions
        self.assertIsNone(result)
        mock_st.error.assert_called_once()

    @patch('zipfile.ZipFile')
    @patch('app.st')
    def test_setup_dataset_from_zip_exception(self, mock_st, mock_zip):
        """Test handling of an invalid or corrupt zip file."""
        mock_zip.side_effect = Exception("Bad Zip File")
        mock_file = MagicMock()
        
        result = setup_dataset_from_zip(mock_file)
        
        self.assertFalse(result)
        mock_st.error.assert_called_once()

    @patch('app.st')
    def test_main_inference_mode(self, mock_st):
        """Test UI switching to Inference mode."""
        from app import main
        # Simulate user selecting "Run System (Inference)" in the sidebar
        mock_st.sidebar.selectbox.return_value = "Run System (Inference)"
        
        main()
        
        # Verify the correct title was rendered
        mock_st.title.assert_called_with(" Inference Mode")

    @patch('app.subprocess.Popen')
    @patch('app.st')
    def test_main_launch_annotation_tool(self, mock_st, mock_popen):
        """Test the desktop annotation tool launch button."""
        from app import main
        
        # Setup Streamlit UI state
        mock_st.sidebar.selectbox.return_value = "Train New Model"
        mock_st.radio.return_value = "Draw Custom Dataset"
        
        # Simulate clicking only the "Launch Annotation Tool" button
        def mock_button_click(label, **kwargs):
            return label == " Launch Annotation Tool"
        mock_st.button.side_effect = mock_button_click
        
        main()
        
        # Verify subprocess was called to launch bbox.py
        mock_popen.assert_called_once()
        mock_st.toast.assert_called_with("Opening Bounding Box Tool...")

    @patch('app.st')
    def test_main_load_custom_dataset_success(self, mock_st):
        """Test successfully loading a custom dataset folder in the UI."""
        from app import main
        
        # Create a mock valid dataset directory
        custom_dir = "mock_custom_dataset"
        os.makedirs(custom_dir, exist_ok=True)
        with open(os.path.join(custom_dir, "dataset.yaml"), "w") as f:
            f.write("mock config")

        # Setup Streamlit UI state
        mock_st.sidebar.selectbox.return_value = "Train New Model"
        mock_st.radio.return_value = "Draw Custom Dataset"
        mock_st.text_input.return_value = custom_dir
        
        # Simulate clicking only the "Load Custom Dataset" button
        def mock_button_click(label, **kwargs):
            return label == "Load Custom Dataset"
        mock_st.button.side_effect = mock_button_click
        
        # Initialize session state mock
        mock_st.session_state = MagicMock()
        
        main()
        
        # Verify success
        self.assertTrue(os.path.exists(self.test_data_dir))
        mock_st.success.assert_called_with("Custom dataset loaded successfully!")
        
        # Cleanup
        shutil.rmtree(custom_dir)

    @patch('app.st')
    def test_main_load_custom_dataset_failure(self, mock_st):
        """Test failing to load an invalid custom dataset folder in the UI."""
        from app import main
        
        # Setup Streamlit UI state with a fake path
        mock_st.sidebar.selectbox.return_value = "Train New Model"
        mock_st.radio.return_value = "Draw Custom Dataset"
        mock_st.text_input.return_value = "non_existent_folder"
        
        def mock_button_click(label, **kwargs):
            return label == "Load Custom Dataset"
        mock_st.button.side_effect = mock_button_click
        
        main()
        
        # Verify error handling
        mock_st.error.assert_called_with(" Invalid folder path or missing .yaml file.")

if __name__ == '__main__':
    unittest.main()