# refer to：https://github.com/bowang-lab/MedSAM/blob/main/gui.py, thanks!
# -*- coding: utf-8 -*-
import sys
import time
from PyQt5.QtCore import (Qt, QRect)
from PyQt5.QtGui import (
    QPainter,
    QPixmap,
    QKeySequence,
    QPen,
    QBrush,
    QColor,
    QImage,
)
from PyQt5.QtWidgets import (
    QFileDialog,
    QApplication,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QShortcut,
    QLineEdit,
    QListWidget,
    QSplitter,
    QRadioButton,
    QButtonGroup,
    QGridLayout,
    QLabel,
    QComboBox,
    QScrollArea,
    QProgressBar,
    QMessageBox
)

import numpy as np
from skimage import transform, io
import torch
from torch.nn import functional as F
from PIL import Image
import cv2
import os

from models.MedSAM import sam_model_registry as medsam_model_registry
from models.SAMMed import sam_model_registry as sammed_model_registry
from models.SAM import sam_model_registry as sam_model_registry

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def np2pixmap(np_img):
    """
    Convert numpy array to QPixmap
    """
    height, width, channel = np_img.shape
    bytesPerLine = 3 * width
    qImg = QImage(np_img.data, width, height, bytesPerLine, QImage.Format_RGB888)
    return QPixmap.fromImage(qImg)


@torch.no_grad()
def sam_inference(model, img_embed, box, point, point_labels, height, width):
    """
    Perform inference on the given image and return the segmentation mask
    """
    if box is not None:
        box_torch = torch.as_tensor(box, dtype=torch.float, device=img_embed.device)
        if len(box_torch.shape) == 2:
            box_torch = box_torch[:, None, :]

        sparse_embeddings, dense_embeddings = model.prompt_encoder(
            points=None,
            boxes=box_torch,
            masks=None,
        )
    elif point is not None:
        coords_torch = torch.as_tensor(point, dtype=torch.float, device=img_embed.device)
        labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=img_embed.device)
        coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]

        points_torch = (coords_torch, labels_torch)

        sparse_embeddings, dense_embeddings = model.prompt_encoder(
            points=points_torch,
            boxes=None,
            masks=None,
        )

    low_res_logits, _ = model.mask_decoder(
        image_embeddings=img_embed,
        image_pe=model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )

    low_res_pred = torch.sigmoid(low_res_logits)

    low_res_pred = F.interpolate(
        low_res_pred,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )
    low_res_pred = low_res_pred.squeeze().cpu().numpy()
    medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
    return medsam_seg


def draw_masks(img, masks):
    """
    Draw masks on the given image
    """
    maskThreshold = 0.5
    if len(img.shape) == 2 or img.shape[2] == 1:
        img = np.stack((img,) * 3, axis=-1)
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]

    for i, mask in enumerate(masks):
        mask = (mask > maskThreshold).astype(np.uint8)
        mask_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        color = colors[i % len(colors)]
        cv2.drawContours(img, mask_contours, -1, color, 1, cv2.LINE_8)

    return img


colors = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 0, 0),
    (0, 128, 0),
    (0, 0, 128),
    (128, 128, 0),
    (128, 0, 128),
    (0, 128, 128),
    (255, 255, 255),
    (192, 192, 192),
    (64, 64, 64),
    (255, 0, 255),
    (0, 255, 255),
    (255, 255, 0),
    (0, 0, 127),
    (192, 0, 192),
]


class Window(QWidget):
    """
    Main window class
    """

    def __init__(self):
        super().__init__()
        self.initializeUI()

    def initializeUI(self):
        self.setupAttributes()
        self.setupLayout()
        self.setupConnections()

    def setupAttributes(self):
        """
        Setup window attributes
        """
        self.half_point_size = 5
        self.annotation_info_file = "annotation_info_file.txt"
        self.image_path = None
        self.color_idx = 0
        self.bg_img = None
        self.is_mouse_down = False
        self.rect = None
        self.point_size = self.half_point_size * 2
        self.start_point = None
        self.end_point = None
        self.start_pos = (None, None)
        self.GT_np = None

        self.x = None
        self.y = None
        self.xmin = None
        self.ymin = None
        self.xmax = None
        self.ymax = None

        self.embedding_sammed_256 = None
        self.embedding_sam_1024 = None
        self.embedding_medsam_1024 = None
        self.sam_model = None
        self.medsam_model = None
        self.sammed_model = None
        self.sam_mask = None
        self.medsam_mask = None
        self.sammed_mask = None

    def setupLayout(self):
        """
        Setup the layout of the window
        """
        main_layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        self.menu_list = QListWidget()
        splitter.addWidget(self.menu_list)

        mid_side_widget = QWidget()
        mid_side_layout = QVBoxLayout()
        mid_side_widget.setLayout(mid_side_layout)
        splitter.addWidget(mid_side_widget)

        right_side_widget = QWidget()
        right_side_layout = QVBoxLayout()
        right_side_widget.setLayout(right_side_layout)
        splitter.addWidget(right_side_widget)

        self.setupViews(mid_side_layout)

        self.setupRadioButtons(mid_side_layout)

        self.setupLoadButtons(mid_side_layout)

        self.setupResultView(mid_side_layout)

        self.setupCoordArea(mid_side_layout)

        self.setupRunAndResultArea(right_side_layout)

        self.setLayout(main_layout)

    def setupRunAndResultArea(self, layout):
        """
        Setup the run button and result display area
        """

        hbox = QHBoxLayout()
        self.run_button = QPushButton("Run")
        self.save_button = QPushButton("Save Masks")
        hbox.addWidget(self.run_button)
        hbox.addWidget(self.save_button)
        layout.addLayout(hbox)

        self.save_button.setEnabled(False)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setGeometry(QRect(30, 40, 200, 25))
        self.progress_bar.setMaximum(100)
        self.progress_bar.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.progress_bar)

        self.gridLayoutWidget = QWidget()
        self.result_display_layout = QGridLayout(self.gridLayoutWidget)

        label = QLabel("Ground Truth")
        self.result_display_layout.addWidget(label, 0, 0)

        fixedWidth = 300
        fixedHeight = 300

        self.seg_view_1 = QGraphicsView()
        self.seg_view_1.setFixedSize(fixedWidth, fixedHeight)
        self.result_display_layout.addWidget(self.seg_view_1, 1, 0)

        self.seg_view_2 = QGraphicsView()
        self.seg_view_2.setFixedSize(fixedWidth, fixedHeight)
        self.result_display_layout.addWidget(self.seg_view_2, 1, 1)

        scrollArea = QScrollArea()
        scrollArea.setWidgetResizable(True)
        scrollArea.setWidget(self.gridLayoutWidget)
        layout.addWidget(scrollArea)

    def setupViews(self, layout):
        """
        Setup the image and result views
        """
        self.view = QGraphicsView()
        self.view.setRenderHint(QPainter.Antialiasing)
        layout.addWidget(self.view)

    def setupResultView(self, layout):
        """
        Setup the result view
        """
        self.result_view = QGraphicsView()
        self.result_view.setRenderHint(QPainter.Antialiasing)
        layout.addWidget(self.result_view)

    def setupRadioButtons(self, layout):
        """
        Setup the radio buttons
        """
        radio_layout = QHBoxLayout()
        self.box_radio = QRadioButton("Box")
        self.point_radio = QRadioButton("Point")
        radio_layout.addWidget(self.box_radio)
        radio_layout.addWidget(self.point_radio)
        self.box_radio.setChecked(True)
        layout.addLayout(radio_layout)

        self.setupGTRadioButtons(layout)

    def setupCoordArea(self, layout):
        """
        Setup the coordinate area
        """
        self.pos_coords = QLineEdit(self)
        coord_layout = QHBoxLayout()
        coord_layout.addWidget(self.pos_coords)
        layout.addLayout(coord_layout)

    def setupGTRadioButtons(self, layout):
        """
        Setup the radio buttons for ground truth
        """
        gt_radio_layout = QHBoxLayout()
        self.gt_radio_group = QButtonGroup(self)
        self.with_gt_radio = QRadioButton("w. gt")
        self.without_gt_radio = QRadioButton("w/o gt")
        gt_radio_layout.addWidget(self.with_gt_radio)
        gt_radio_layout.addWidget(self.without_gt_radio)
        self.without_gt_radio.setChecked(True)
        layout.addLayout(gt_radio_layout)

        self.with_gt_radio.setEnabled(False)
        self.without_gt_radio.setChecked(True)
        self.gt_radio_group.addButton(self.with_gt_radio)
        self.gt_radio_group.addButton(self.without_gt_radio)

    def setupLoadButtons(self, layout):
        """
        Setup the load image and load GT buttons
        """
        hbox = QHBoxLayout()
        self.load_image_button = QPushButton("Load Image")
        self.load_GT_button = QPushButton("Load GT")
        hbox.addWidget(self.load_image_button)
        hbox.addWidget(self.load_GT_button)
        layout.addLayout(hbox)

        self.load_GT_button.setEnabled(False)

    def setupConnections(self):
        """
        Setup the connections of the widgets
        """
        self.run_button.clicked.connect(self.run_segmentation_model)
        self.save_button.clicked.connect(self.save_masks)
        self.load_image_button.clicked.connect(self.load_image)
        self.load_GT_button.clicked.connect(self.load_GT)
        self.menu_list.itemClicked.connect(self.display_result)
        self.with_gt_radio.toggled.connect(self.handle_with_gt_radio_toggled)

    def resetState(self):
        """
        Reset the state of the window
        """
        self.image_path = None
        self.image_name = None
        self.bg_img = None
        self.img_3c = None
        self.display_img = None

        self.start_point = None
        self.end_point = None
        self.rect = None
        self.is_mouse_down = False

        self.x = None
        self.y = None
        self.xmin = None
        self.ymin = None
        self.xmax = None
        self.ymax = None

        self.embedding_sammed_256 = None
        self.embedding_sam_1024 = None
        self.embedding_medsam_1024 = None
        self.sam_model = None
        self.medsam_model = None
        self.sammed_model = None
        self.sam_mask = None
        self.medsam_mask = None
        self.sammed_mask = None

        self.menu_list.clear()
        self.pos_coords.clear()

        if self.result_view.scene():
            self.result_view.scene().clear()

        if self.seg_view_1.scene():
            self.seg_view_1.scene().clear()

        if self.seg_view_2.scene():
            self.seg_view_2.scene().clear()

        self.with_gt_radio.setEnabled(False)

    def load_image(self):
        """
        Load the image to segment
        """
        self.resetState()

        file_path, file_type = QFileDialog.getOpenFileName(
            self, "Choose Image to Segment", ".", "Image Files (*.png *.jpg *.bmp)"
        )

        if file_path is None or len(file_path) == 0:
            print("No image path specified, plz select an image")
            exit()

        img_np = io.imread(file_path)
        if len(img_np.shape) == 2:
            img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
        else:
            img_3c = img_np

        self.img_3c = img_3c
        self.display_img = img_3c.copy()
        self.image_path = file_path
        self.image_name = os.path.basename(file_path).split(".")[0]
        pixmap = np2pixmap(self.img_3c)

        H, W, _ = self.img_3c.shape

        self.scene = QGraphicsScene(0, 0, W, H)
        self.end_point = None
        self.rect = None
        self.bg_img = self.scene.addPixmap(pixmap)
        self.bg_img.setPos(0, 0)
        self.view.setScene(self.scene)

        self.check_and_draw(self.image_name)

        self.scene.mousePressEvent = self.mouse_press
        self.scene.mouseMoveEvent = self.mouse_move
        self.scene.mouseReleaseEvent = self.mouse_release

        self.load_GT_button.setEnabled(True)

    def load_GT(self):
        """
        Load the ground truth image
        """
        file_path, file_type = QFileDialog.getOpenFileName(
            self, "Choose Image to Segment", ".", "Image Files (*.png *.jpg *.bmp)"
        )

        if file_path is None or len(file_path) == 0:
            print("No image path specified, plz select an image")
            exit()

        self.GT_np = io.imread(file_path)
        self.GT_3c = np.repeat(self.GT_np[:, :, None], 3, axis=-1)
        self.GT_3c_green = self.GT_3c.copy()
        self.GT_3c_green[self.GT_np != 0] = colors[1]
        scene = QGraphicsScene(0, 0, self.GT_3c.shape[0], self.GT_3c.shape[1])
        scene.addPixmap(np2pixmap(self.GT_3c))
        self.seg_view_2.setScene(scene)

        bg = Image.fromarray(self.img_3c.astype("uint8"), "RGB")
        mask = Image.fromarray(self.GT_3c_green.astype("uint8"), "RGB")
        img = Image.blend(bg, mask, 0.2)
        scene = QGraphicsScene(0, 0, self.GT_3c.shape[0], self.GT_3c.shape[1])
        scene.addPixmap(np2pixmap(np.array(img)))
        self.seg_view_1.setScene(scene)

        self.with_gt_radio.setEnabled(True)
        self.with_gt_radio.setChecked(True)
        self.handle_with_gt_radio_toggled(True)

    def handle_with_gt_radio_toggled(self, checked):
        """
        Handle the toggling of the radio button for ground truth
        """
        if checked and hasattr(self, 'GT_np'):
            self.display_img = draw_masks(self.img_3c.copy(), [self.GT_np])
        else:
            self.display_img = self.img_3c.copy()

        self.update_view_with_image(self.display_img)

    def update_view_with_image(self, np_img):
        """
        Update the view with the given image
        """
        pixmap = np2pixmap(np_img)

        if self.bg_img is None:
            self.bg_img = self.scene.addPixmap(pixmap)
            self.bg_img.setPos(0, 0)
        else:
            self.bg_img.setPixmap(pixmap)

    def check_and_draw(self, image_name):
        """
        Check if the image has been annotated before and draw the annotations
        """
        if not os.path.exists(self.annotation_info_file):
            return

        with open(self.annotation_info_file, "r") as f:
            for line in f.readlines():
                saved_image_name, coords_str, timestamp = line.strip().split("; ")
                if saved_image_name == image_name:
                    if len(eval(coords_str)) == 4:
                        xmin, ymin, xmax, ymax = eval(coords_str)
                        self.menu_list.addItem(f"({str(xmin)}, {str(ymin)}, {str(xmax)}, {str(ymax)}) at {timestamp}")
                    elif len(eval(coords_str)) == 2:
                        x, y = eval(coords_str)
                        self.menu_list.addItem(f"({str(x)}, {str(y)}) at {timestamp}")

    def mouse_press(self, ev):
        """
        Handle the mouse press event
        """
        x, y = ev.scenePos().x(), ev.scenePos().y()
        self.is_mouse_down = True
        self.start_pos = ev.scenePos().x(), ev.scenePos().y()
        self.start_point = self.scene.addEllipse(
            x - self.half_point_size,
            y - self.half_point_size,
            self.point_size,
            self.point_size,
            pen=QPen(QColor("red")),
            brush=QBrush(QColor("red")),
        )

    def mouse_move(self, ev):
        """
        Handle the mouse move event
        """
        if not self.is_mouse_down:
            return

        x, y = ev.scenePos().x(), ev.scenePos().y()

        if self.end_point is not None:
            self.scene.removeItem(self.end_point)
        self.end_point = self.scene.addEllipse(
            x - self.half_point_size,
            y - self.half_point_size,
            self.point_size,
            self.point_size,
            pen=QPen(QColor("red")),
            brush=QBrush(QColor("red")),
        )

        if self.rect is not None:
            self.scene.removeItem(self.rect)
        sx, sy = self.start_pos
        xmin = min(x, sx)
        xmax = max(x, sx)
        ymin = min(y, sy)
        ymax = max(y, sy)
        self.rect = self.scene.addRect(
            xmin, ymin, xmax - xmin, ymax - ymin, pen=QPen(QColor("red"))
        )

    def mouse_release(self, ev):
        """
        Handle the mouse release event
        """
        self.xmin = None
        self.ymin = None
        self.xmax = None
        self.ymax = None
        self.x = None
        self.y = None

        timestamp = time.strftime("%H:%M:%S")
        self.is_mouse_down = False

        H, W, _ = self.img_3c.shape

        if self.box_radio.isChecked():
            (xmin, ymin, xmax, ymax), mask_with_box = self.handle_box_release(ev)
            self.pos_coords.setText(f"({str(xmin)}, {str(ymin)}, {str(xmax)}, {str(ymax)})")
            self.save_annotation_info(self.image_name, (xmin, ymin, xmax, ymax), timestamp)

        elif self.point_radio.isChecked():
            (x, y), mask_with_box = self.handle_point_release(ev)
            self.pos_coords.setText(f"({str(x)}, {str(y)})")
            self.save_annotation_info(self.image_name, (x, y), timestamp)

        self.scene.removeItem(self.bg_img)
        self.bg_img = self.scene.addPixmap(np2pixmap(mask_with_box))

        result_scene = QGraphicsScene(0, 0, W, H)
        result_pixmap = np2pixmap(mask_with_box)
        result_scene.addPixmap(result_pixmap)
        self.result_view.setScene(result_scene)

    def handle_box_release(self, ev):
        """
        Handle the box release event
        """
        x, y = ev.scenePos().x(), ev.scenePos().y()
        sx, sy = self.start_pos
        xmin = min(x, sx)
        xmax = max(x, sx)
        ymin = min(y, sy)
        ymax = max(y, sy)

        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

        box_np = np.array([[xmin, ymin, xmax, ymax]])
        mask_with_box = self.display_img.copy()
        cv2.rectangle(mask_with_box, (int(xmin), int(ymin)), (int(xmax), int(ymax)),
                      colors[0], 2)

        return (xmin, ymin, xmax, ymax), mask_with_box

    def handle_point_release(self, ev):
        """
        Handle the point release event
        """
        x, y = ev.scenePos().x(), ev.scenePos().y()

        self.is_mouse_down = False

        self.x = x
        self.y = y

        mask_with_box = self.display_img.copy()
        cv2.circle(mask_with_box, (int(x), int(y)), 5, colors[0], 2)

        return (x, y), mask_with_box

    def display_result(self, item):
        """
        Display the result of the selected annotation
        """
        self.xmin = None
        self.ymin = None
        self.xmax = None
        self.ymax = None
        self.x = None
        self.y = None

        coord_str = item.text().split(" at ")[0]
        result_image = self.display_img.copy()
        if len(eval(coord_str)) == 4:
            xmin, ymin, xmax, ymax = eval(coord_str)

            self.xmin = xmin
            self.ymin = ymin
            self.xmax = xmax
            self.ymax = ymax

            cv2.rectangle(result_image, (int(xmin), int(ymin)), (int(xmax), int(ymax)),
                          colors[0], 2)
            self.pos_coords.setText(f"({str(xmin)}, {str(ymin)}, {str(xmax)}, {str(ymax)})")

        elif len(eval(coord_str)) == 2:
            x, y = eval(coord_str)

            self.x = x
            self.y = y

            cv2.circle(result_image, (int(x), int(y)), 5, colors[0], 2)
            self.pos_coords.setText(f"({str(x)}, {str(y)})")

        result_scene = QGraphicsScene(0, 0, result_image.shape[1], result_image.shape[0])
        result_pixmap = np2pixmap(result_image)
        result_scene.addPixmap(result_pixmap)

        self.result_view.setScene(result_scene)

    def save_annotation_info(self, image_name, coords, timestamp):
        """
        Save the annotation info to the file
        """
        if len(coords) == 4:
            xmin, ymin, xmax, ymax = coords
            self.menu_list.addItem(f"({str(xmin)}, {str(ymin)}, {str(xmax)}, {str(ymax)}) at {timestamp}")
        if len(coords) == 2:
            x, y = coords
            self.menu_list.addItem(f"({str(x)}, {str(y)}) at {timestamp}")
        with open(self.annotation_info_file, "a") as f:
            f.write(f"{image_name}; {coords}; {timestamp}\n")

    def image_preprocess(self, size):
        """
        Preprocess the image for inference
        """
        img = transform.resize(
            self.img_3c.copy(), (size, size), order=3, preserve_range=True, anti_aliasing=True
        ).astype(np.uint8)
        img = (img - img.min()) / np.clip(
            img.max() - img.min(), a_min=1e-8, a_max=None
        )
        img_tensor = (
            torch.tensor(img).float().permute(2, 0, 1).unsqueeze(0).to(device)
        )

        return img_tensor

    def process_segmentation(self, model, model_name, embedding_size, embedding, color_index, layout_row, layout_col,
                             label_text):
        """
        Process the segmentation result
        """
        H, W, _ = self.img_3c.shape

        if self.x is not None:
            point_np = np.array([(self.x, self.y)])
            point_or_box_scaled = point_np / np.array([W, H]) * embedding_size
            mask = sam_inference(model, embedding, None, point_or_box_scaled, np.array([1]), H, W)
        elif self.xmin is not None:
            box_np = np.array([(self.xmin, self.ymin, self.xmax, self.ymax)])
            point_or_box_scaled = box_np / np.array([W, H, W, H]) * embedding_size
            mask = sam_inference(model, embedding, point_or_box_scaled, None, np.array([1]), H, W)

        self.display_segmentation_result(mask, H, W, color_index, layout_row, layout_col, label_text)

        return mask

    def display_segmentation_result(self, mask, H, W, color_index, row, col, label_text):
        """
        Display the segmentation result
        """
        fixedWidth = 300
        fixedHeight = 300

        label = QLabel(label_text)
        self.result_display_layout.addWidget(label, row, col)

        mask_3c = np.repeat(mask[:, :, None], 3, axis=-1)
        mask_3c_color = mask_3c.copy()
        mask_3c_color[mask != 0] = colors[color_index]

        seg_view_fusion = QGraphicsView()
        seg_view_fusion.setFixedSize(fixedWidth, fixedHeight)
        self.result_display_layout.addWidget(seg_view_fusion, row + 1, col)

        bg = Image.fromarray(self.img_3c.astype("uint8"), "RGB")
        mask_img = Image.fromarray(mask_3c_color.astype("uint8"), "RGB")
        blended_img = Image.blend(bg, mask_img, 0.2)

        scene = QGraphicsScene(0, 0, H, W)
        scene.addPixmap(np2pixmap(np.array(blended_img)))
        seg_view_fusion.setScene(scene)

        seg_view_mask = QGraphicsView()
        seg_view_mask.setFixedSize(fixedWidth, fixedHeight)
        self.result_display_layout.addWidget(seg_view_mask, row + 1, col + 1)

        scene = QGraphicsScene(0, 0, H, W)
        scene.addPixmap(np2pixmap(mask_3c * 255))
        seg_view_mask.setScene(scene)

    def clear_layout(self, layout, from_row=2):
        """
        Clear the layout
        """
        for i in reversed(range(layout.count())):
            widget = layout.itemAt(i).widget()
            if widget is not None:
                if layout.getItemPosition(i)[0] >= from_row:
                    layout.removeWidget(widget)
                    widget.deleteLater()
            else:
                layout_item = layout.itemAt(i)
                if layout_item is not None:
                    self.clear_layout(layout_item.layout(), from_row)

    def run_segmentation_model(self):
        """
        Run the segmentation model
        """

        if self.x == None and self.xmin == None:
            QMessageBox.warning(self, "Warning", "Please provide a bounding box or point to segment.")
            return

        steps = 6
        step = 0
        self.clear_layout(self.result_display_layout)

        if self.sam_model == None:
            self.sam_model = sam_model_registry["vit_b"](checkpoint='./checkpoints/SAM/sam_vit_b_01ec64.pth').to(device)
            self.sam_model.eval()

        self.progress_bar.setValue(int((step / steps) * 100))
        step += 1

        if self.embedding_sam_1024 == None:
            img_tensor = self.image_preprocess(1024)
            with torch.no_grad():
                self.embedding_sam_1024 = self.sam_model.image_encoder(
                    img_tensor
                )

        self.progress_bar.setValue(int((step / steps) * 100))
        step += 1

        if self.medsam_model == None:
            self.medsam_model = medsam_model_registry["vit_b"](checkpoint='./checkpoints/MedSAM/medsam_vit_b.pth').to(
                device)
            self.medsam_model.eval()

        self.progress_bar.setValue(int((step / steps) * 100))
        step += 1

        if self.embedding_medsam_1024 == None:
            img_tensor = self.image_preprocess(1024)
            with torch.no_grad():
                self.embedding_medsam_1024 = self.medsam_model.image_encoder(
                    img_tensor
                )

        self.progress_bar.setValue(int((step / steps) * 100))
        step += 1

        if self.sammed_model == None:
            self.sammed_model = sammed_model_registry["vit_b"](checkpoint='./checkpoints/SAMMed/sam-med2d_b.pth').to(
                device)
            self.sammed_model.eval()

        self.progress_bar.setValue(int((step / steps) * 100))
        step += 1

        if self.embedding_sammed_256 == None:
            img_tensor = self.image_preprocess(256)
            with torch.no_grad():
                self.embedding_sammed_256 = self.sammed_model.image_encoder(
                    img_tensor
                )

        self.progress_bar.setValue(int((step / steps) * 100))
        step += 1

        self.progress_bar.setValue(100)

        self.sam_mask = self.process_segmentation(self.sam_model, "sam", 1024, self.embedding_sam_1024, 0, 2, 0,
                                                  "SAM Model")
        self.medsam_mask = self.process_segmentation(self.medsam_model, "medsam", 1024, self.embedding_medsam_1024, 0,
                                                     4, 0,
                                                     "MedSAM Model")
        self.sammed_mask = self.process_segmentation(self.sammed_model, "sammed", 256, self.embedding_sammed_256, 0, 6,
                                                     0, "SAMMed Model")

        self.save_button.setEnabled(True)

    def save_masks(self):
        """
        Save the masks to disk
        """
        if not os.path.isdir("./results"):
            os.makedirs("./results")

        if self.x is not None and self.y is not None:
            dir_path = f"./results/{self.image_name}_point_{self.x}_{self.y}"
            if not os.path.isdir(dir_path):
                os.makedirs(dir_path)

            # Save the masks and images
            Image.fromarray(self.img_3c).save(os.path.join(dir_path, f"{self.image_name}.png"))
            if self.GT_np != None:
                Image.fromarray(self.GT_np).save(os.path.join(dir_path, f"{self.image_name}_GT.png"))
            Image.fromarray(self.sam_mask * 255).save(os.path.join(dir_path, f"{self.image_name}_sam_mask.png"))
            Image.fromarray(self.medsam_mask * 255).save(os.path.join(dir_path, f"{self.image_name}_medsam_mask.png"))
            Image.fromarray(self.sammed_mask * 255).save(os.path.join(dir_path, f"{self.image_name}_sammed_mask.png"))

        elif self.xmin is not None and self.ymin is not None and self.xmax is not None and self.ymax is not None:
            # Create a directory for bounding box-based segmentation
            dir_path = f"./results/{self.image_name}_bbox_{self.xmin}_{self.ymin}_{self.xmax}_{self.ymax}"
            if not os.path.isdir(dir_path):
                os.makedirs(dir_path)

            # Save the masks and images
            Image.fromarray(self.img_3c).save(os.path.join(dir_path, f"{self.image_name}.png"))
            if self.GT_np != None:
                Image.fromarray(self.GT_np).save(os.path.join(dir_path, f"{self.image_name}_GT.png"))
            Image.fromarray(self.sam_mask * 255).save(os.path.join(dir_path, f"{self.image_name}_sam_mask.png"))
            Image.fromarray(self.medsam_mask * 255).save(os.path.join(dir_path, f"{self.image_name}_medsam_mask.png"))
            Image.fromarray(self.sammed_mask * 255).save(os.path.join(dir_path, f"{self.image_name}_sammed_mask.png"))

        QMessageBox.information(self, "Success", "Saved.")


app = QApplication(sys.argv)

w = Window()
w.show()

app.exec()
