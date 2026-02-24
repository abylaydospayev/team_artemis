import sys
import os
from httpx import delete
import yaml
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton,
    QFileDialog, QWidget, QVBoxLayout, QHBoxLayout,
    QListWidget, QListWidgetItem, QLineEdit, QMessageBox,
    QComboBox, QSpinBox, QGroupBox, QSplitter
)
from PySide6.QtGui import QPixmap, QPainter, QPen, QColor, QFont
from PySide6.QtCore import Qt, QRect, QPoint

from PIL import Image


class ImageLabel(QLabel):
    def __init__(self):
        super().__init__()
        self.start = None
        self.end = None
        self.boxes = []  # List of (class_id, rect) tuples
        self.current_class = 0
        self.class_colors = [
            QColor(255, 0, 0),    # red
            QColor(0, 255, 0),    # green
            QColor(0, 0, 255),    # blue
            QColor(255, 255, 0),  # yellow
            QColor(255, 0, 255),  # magenta
            QColor(0, 255, 255),  # cyan
        ]
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(640, 480)
        self.selected_box_index = -1

    def set_current_class(self, class_id):
        self.current_class = class_id

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.start = e.position().toPoint()
            self.end = self.start
            self.update()
        elif e.button() == Qt.RightButton:
            # Right-click to select boxes
            pos = e.position().toPoint()
            self.select_box_at_position(pos)

    def select_box_at_position(self, pos):
        for i, (_, rect) in enumerate(self.boxes):
            if rect.contains(pos):
                self.selected_box_index = i
                self.update()
                return
        self.selected_box_index = -1
        self.update()

    def delete_selected_box(self):
        if self.selected_box_index >= 0 and self.selected_box_index < len(self.boxes):
            del self.boxes[self.selected_box_index]
            self.selected_box_index = -1
            self.update()

    def mouseMoveEvent(self, e):
        if self.start:
            self.end = e.position().toPoint()
            self.update()

    def mouseReleaseEvent(self, e):
        if self.start and self.end and self.start != self.end:
            rect = QRect(self.start, self.end).normalized()
            # Check if rectangle has minimum size
            if rect.width() > 5 and rect.height() > 5:
                self.boxes.append((self.current_class, rect))
            else:
                QMessageBox.warning(self, "Small Box", "Box too small, please draw a larger box")

        self.start = None
        self.end = None
        self.update()

    def paintEvent(self, e):
        super().paintEvent(e)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Draw existing boxes
        for i, (class_id, box) in enumerate(self.boxes):
            color = self.class_colors[class_id % len(self.class_colors)]
            pen = QPen(color, 2)
            
            # Highlight selected box
            if i == self.selected_box_index:
                pen.setWidth(4)
                pen.setStyle(Qt.DashLine)
            
            painter.setPen(pen)
            painter.drawRect(box)
            
            # Draw class label
            painter.setPen(QPen(color.darker(), 1))
            painter.setBrush(color)
            painter.drawRect(box.left(), box.top() - 20, 50, 20)
            painter.setPen(QPen(Qt.white))
            painter.drawText(box.left() + 5, box.top() - 5, f"Class {class_id}")

        # Draw current box being drawn
        if self.start and self.end:
            painter.setPen(QPen(Qt.green, 2, Qt.DashLine))
            painter.drawRect(QRect(self.start, self.end).normalized())

    def clear_boxes(self):
        self.boxes.clear()
        self.selected_box_index = -1
        self.update()


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Bounding Box Drawing Tool")
        self.resize(1200, 800)

        # dataStorage
        self.images = []
        self.index = 0
        self.classes = ["weed", "crop"]  # example classes.

        self.training_folder = ""
        self.label_folder = "labels"
        self.image_folder = ""
        
        # UI setup
        self.setup_ui()

    def setup_ui(self):
        # main widget and layout
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        
        # create splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        
        # left panel for image annotation
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        
        # image label
        self.label = ImageLabel()
        
        # image navigation
        nav_layout = QHBoxLayout()
        self.prev_btn = QPushButton("← Prev")
        self.next_btn = QPushButton("Next →")
        self.prev_btn.clicked.connect(self.prev_image)
        self.next_btn.clicked.connect(self.next_image)
        nav_layout.addWidget(self.prev_btn)
        nav_layout.addWidget(self.next_btn)
        
        # image not load
        self.image_info_label = QLabel("No image loaded")
        self.image_info_label.setAlignment(Qt.AlignCenter)
        
        left_layout.addWidget(self.label)
        left_layout.addWidget(self.image_info_label)
        left_layout.addLayout(nav_layout)
        left_panel.setLayout(left_layout)
        
        # right panel for the controls
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        right_layout.setAlignment(Qt.AlignTop)
        
        # pipeline steps group
        pipeline_group = QGroupBox("Pipeline Steps")
        pipeline_layout = QVBoxLayout()
        
        # 1 tutorial
        self.create_model_btn = QPushButton("1. Tutorial")
        self.create_model_btn.clicked.connect(self.create_new_model)
        pipeline_layout.addWidget(self.create_model_btn)
        
        # 2 select folders
        folder_layout = QHBoxLayout()
        self.select_image_folder_btn = QPushButton("2.a Select Image Folder")
        self.select_image_folder_btn.clicked.connect(self.open_folder)
        self.select_label_folder_btn = QPushButton("2.b Select Label Folder")
        self.select_label_folder_btn.clicked.connect(self.select_label_folder)
        folder_layout.addWidget(self.select_image_folder_btn)
        folder_layout.addWidget(self.select_label_folder_btn)
        pipeline_layout.addLayout(folder_layout)
        
        # 3 check data loaded or not
        self.start_training_btn = QPushButton("3. Start Training (Prepare Data)")
        self.start_training_btn.clicked.connect(self.prepare_training_data)
        pipeline_layout.addWidget(self.start_training_btn)
        
        # 4 workspace section
        self.workspace_label = QLabel("4. Box drawing workspace ready")
        # self.workspace_label.setStyleSheet("color: green;")
        pipeline_layout.addWidget(self.workspace_label)
        
        # 5 generate .yaml
        self.generate_yaml_btn = QPushButton("5. Generate YAML File")
        self.generate_yaml_btn.clicked.connect(self.generate_yaml_file)
        pipeline_layout.addWidget(self.generate_yaml_btn)
        
        # 6 training status
        self.training_status = QLabel("6. Dataset ready for training")
        pipeline_layout.addWidget(self.training_status)
        
        # 7 to training page
        self.back_main_btn = QPushButton("7. Back to Main Frame")
        self.back_main_btn.clicked.connect(self.back_to_main)
        pipeline_layout.addWidget(self.back_main_btn)
        
        pipeline_group.setLayout(pipeline_layout)
        right_layout.addWidget(pipeline_group)
        
        # annotation tools group
        annotation_group = QGroupBox("Annotation Tools")
        annotation_layout = QVBoxLayout()
        
        # class section
        class_layout = QHBoxLayout()
        class_layout.addWidget(QLabel("Class:"))
        self.class_combo = QComboBox()
        self.class_combo.addItems(self.classes)
        self.class_combo.currentIndexChanged.connect(self.change_class)
        class_layout.addWidget(self.class_combo)
        annotation_layout.addLayout(class_layout)
        
        # add n delete class
        new_class_layout = QHBoxLayout()
        self.new_class_input = QLineEdit()
        self.new_class_input.setPlaceholderText("New class name")
        add_class_btn = QPushButton("Add Class")
        add_class_btn.clicked.connect(self.add_new_class)
        delete_class_btn = QPushButton("Delete Current Class")
        delete_class_btn.clicked.connect(self.delete_current_class)
        new_class_layout.addWidget(self.new_class_input)
        new_class_layout.addWidget(add_class_btn)
        new_class_layout.addWidget(delete_class_btn)
        annotation_layout.addLayout(new_class_layout)
        
        # box operations
        box_ops_layout = QHBoxLayout()
        self.delete_box_btn = QPushButton("Delete Selected Box")
        self.delete_box_btn.clicked.connect(self.delete_selected_box)
        self.clear_all_btn = QPushButton("Clear All Boxes")
        self.clear_all_btn.clicked.connect(self.clear_all_boxes)
        box_ops_layout.addWidget(self.delete_box_btn)
        box_ops_layout.addWidget(self.clear_all_btn)
        annotation_layout.addLayout(box_ops_layout)
        
        # save
        self.save_btn = QPushButton("Save Labels")
        self.save_btn.clicked.connect(self.save_labels)
        annotation_layout.addWidget(self.save_btn)
        
        # instructions
        instructions = QLabel(
            "Instructions:\n"
            "• Left click & drag to draw box\n"
            "• Right click on box to select\n"
            "• Select class before drawing\n"
            "• Use Delete Selected to remove"
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet("background-color: #f0f0f0; padding: 10px;")
        annotation_layout.addWidget(instructions)
        
        annotation_group.setLayout(annotation_layout)
        right_layout.addWidget(annotation_group)
        
        # stats
        self.stats_label = QLabel("Images: 0 | Labels: 0")
        right_layout.addWidget(self.stats_label)
        
        right_layout.addStretch()
        right_panel.setLayout(right_layout)
        
        # add panels to splitter left and right
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([800, 400])
        
        main_layout.addWidget(splitter)
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

    def create_new_model(self):
        QMessageBox.information(self, "Create Model", 
            "Model creation tutorial:\n"
            "1. Load your dataset images\n"
            "2. Create desired class names/labels\n"
            "3. Draw bounding boxes on all images in the dataset\n"
            "4. Use 'Generate YAML File' button when ready")
        self.statusBar().showMessage("Ready to create model")

    def select_label_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Label Save Folder"
        )
        if folder:
            self.label_folder = folder
            self.select_label_folder_btn.setText(f"Labels: {os.path.basename(folder)}")
            os.makedirs(self.label_folder, exist_ok=True)

    def prepare_training_data(self):
        if not self.images:
            QMessageBox.warning(self, "No Data", "Please load images first")
            return
        
        # save all current labels
        self.save_labels()
        
        # count total annotations
        total_boxes = 0
        for img_path in self.images:
            name = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(self.label_folder, f"{name}.txt")
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    total_boxes += len(f.readlines())
        
        QMessageBox.information(self, "Data Ready", 
            f"Training data prepared:\n"
            f"Images: {len(self.images)}\n"
            f"Total annotations: {total_boxes}\n"
            f"Labels saved in: {self.label_folder}")
        
        self.start_training_btn.setText("3. Images are loaded")
        self.start_training_btn.setEnabled(False)

    def generate_yaml_file(self):
        if not self.images or not os.path.exists(self.label_folder):
            QMessageBox.warning(self, "Missing Data", "Please prepare data first")
            return
        
        # get save location for .yaml
        yaml_path, _ = QFileDialog.getSaveFileName(
            self, "Save YAML Configuration", "dataset.yaml", "YAML files (*.yaml)"
        )
        
        if yaml_path:
            # yaml content
            data = {
                'path': os.path.dirname(self.images[0]),  # dataset root dir
                'train': 'images',  # train images (relative to 'path')
                'val': 'images',    # val images (relative to 'path')
                'nc': len(self.classes),  # number of classes
                'names': self.classes  # class names
            }
            
            with open(yaml_path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
            
            self.generate_yaml_btn.setText("5. YAML Generated")
            QMessageBox.information(self, "Success", f"YAML file saved to:\n{yaml_path}")

    def back_to_main(self):
        # save current progress
        self.save_labels()
        reply = QMessageBox.question(self, 'Back to Main', 
            'Save progress and return to main frame?',
            QMessageBox.Yes | QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            self.statusBar().showMessage("Returning to main frame...")
            # to main frame place holder

    def add_new_class(self):
        new_class = self.new_class_input.text().strip()
        if new_class and new_class not in self.classes:
            self.classes.append(new_class)
            self.class_combo.addItem(new_class)
            self.new_class_input.clear()
            self.statusBar().showMessage(f"Added new class: {new_class}")

    def delete_current_class(self):
        current_index = self.class_combo.currentIndex()
        if current_index >= 0:
            class_name = self.classes[current_index]
            reply = QMessageBox.question(self, 'Delete Class', 
                f'Are you sure you want to delete class "{class_name}"?',
                QMessageBox.Yes | QMessageBox.No)
            
            if reply == QMessageBox.Yes:
                del self.classes[current_index]
                self.class_combo.removeItem(current_index)
                self.statusBar().showMessage(f"Deleted class: {class_name}")

    def change_class(self, index):
        self.label.set_current_class(index)

    def delete_selected_box(self):
        self.label.delete_selected_box()

    def clear_all_boxes(self):
        self.label.clear_boxes()

    def open_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Image Folder"
        )
        if not folder:
            return

        self.image_folder = folder
        self.images = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith((".jpg", ".png", ".jpeg", ".bmp"))
        ]
        self.images.sort()
        
        if not self.images:
            QMessageBox.warning(self, "No Images", "No supported images found in folder")
            return
        
        self.index = 0
        
        # create label folder if not set
        if self.label_folder == "labels":
            self.label_folder = os.path.join(os.path.dirname(folder), "labels")
        
        os.makedirs(self.label_folder, exist_ok=True)
        
        self.select_image_folder_btn.setText(f"Images: {os.path.basename(folder)}")
        self.load_image()
        self.update_stats()

    def load_image(self):
        if not self.images:
            return

        path = self.images[self.index]
        pixmap = QPixmap(path)
        
        # fit label while maintaining aspect ratio
        scaled_pixmap = pixmap.scaled(
            self.label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.label.setPixmap(scaled_pixmap)
        
        # load existing labels
        self.load_labels()
        
        # update img info
        name = os.path.basename(path)
        self.image_info_label.setText(f"Image {self.index + 1}/{len(self.images)}: {name}")

    def load_labels(self):
        if not self.images:
            return
        
        img_path = self.images[self.index]
        name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(self.label_folder, f"{name}.txt")
        
        self.label.clear_boxes()
        
        if os.path.exists(label_path):
            img = Image.open(img_path)
            w, h = img.size
            
            # get the scale factors between original image and displayed pixmap
            pixmap = self.label.pixmap()
            if pixmap:
                display_w = pixmap.width()
                display_h = pixmap.height()
                
                scale_x = display_w / w
                scale_y = display_h / h
                
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            class_id, xc, yc, bw, bh = map(float, parts)
                            
                            # convert yolo format to pixel coordinates label
                            xc_pix = xc * display_w
                            yc_pix = yc * display_h
                            bw_pix = bw * display_w
                            bh_pix = bh * display_h
                            
                            x1 = int(xc_pix - bw_pix/2)
                            y1 = int(yc_pix - bh_pix/2)
                            x2 = int(xc_pix + bw_pix/2)
                            y2 = int(yc_pix + bh_pix/2)
                            
                            rect = QRect(x1, y1, x2 - x1, y2 - y1)
                            self.label.boxes.append((int(class_id), rect))

    def save_labels(self):
        if not self.images:
            return

        img_path = self.images[self.index]
        name = os.path.splitext(os.path.basename(img_path))[0]
        
        # get original image dimensions
        img = Image.open(img_path)
        orig_w, orig_h = img.size
        
        # get displayed pixmap dimensions
        pixmap = self.label.pixmap()
        if not pixmap:
            return
        
        display_w = pixmap.width()
        display_h = pixmap.height()
        
        # calculate scale factors
        scale_x = orig_w / display_w
        scale_y = orig_h / display_h
        
        label_path = os.path.join(self.label_folder, f"{name}.txt")
        
        with open(label_path, "w") as f:
            for class_id, box in self.label.boxes:
                # convert display coordinates to original image coordinates
                x1_disp = box.left()
                y1_disp = box.top()
                x2_disp = box.right()
                y2_disp = box.bottom()
                
                # scale to original image size
                x1 = x1_disp * scale_x
                y1 = y1_disp * scale_y
                x2 = x2_disp * scale_x
                y2 = y2_disp * scale_y
                
                # convert to yolo format
                xc = ((x1 + x2) / 2) / orig_w
                yc = ((y1 + y2) / 2) / orig_h
                bw = (x2 - x1) / orig_w
                bh = (y2 - y1) / orig_h
                
                # values are within [0, 1]
                xc = max(0, min(1, xc))
                yc = max(0, min(1, yc))
                bw = max(0, min(1, bw))
                bh = max(0, min(1, bh))
                
                f.write(f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
        
        self.statusBar().showMessage(f"Saved labels to {label_path}")
        self.update_stats()

    def next_image(self):
        if self.index < len(self.images) - 1:
            self.save_labels()
            self.index += 1
            self.load_image()

    def prev_image(self):
        if self.index > 0:
            self.save_labels()
            self.index -= 1
            self.load_image()

    def update_stats(self):
        if not self.images:
            self.stats_label.setText("Images: 0 | Labels: 0")
            return
        
        # count total labels
        total_labels = 0
        for img_path in self.images:
            name = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(self.label_folder, f"{name}.txt")
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    total_labels += len(f.readlines())
        
        self.stats_label.setText(f"Images: {len(self.images)} | Total Labels: {total_labels}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())