import shutil
import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QPushButton, QLabel, QListWidget, QFileDialog,
                             QProgressBar, QCheckBox, QInputDialog, QHBoxLayout,
                             QScrollArea, QFrame, QMessageBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from datetime import datetime
from PyQt5.QtGui import QDragEnterEvent, QDropEvent
import pydicom
from skimage.draw import polygon
import os
import re
import json
from collections import OrderedDict
import SimpleITK as sitk
import numpy as np
import subprocess

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def process_subfolders(patient_path):
    # 去重
    try:
        subfolders = [
            f for f in os.listdir(patient_path)
            if os.path.isdir(os.path.join(patient_path, f))
        ]
    except Exception as e:
        print(f"获取子文件夹失败: {e}")
        return False

    if len(subfolders) == 1:
        print("无需处理,子文件夹数量为1")
        return False

    # 无有效日期，按字符串字典序降序排序
    subfolders.sort(reverse=True)
    latest_folder = subfolders[0]
    print(f"未检测到标准日期，按名称排序保留: {latest_folder}")

    # 删除 && 保留
    deleted_count = 0
    for folder in subfolders:
        if folder == latest_folder:
            continue  # 跳过最新文件夹

        folder_path_to_delete = os.path.join(patient_path, folder)
        try:
            shutil.rmtree(folder_path_to_delete)
            deleted_count += 1
            print(f"已删除: {folder_path_to_delete}")
        except Exception as e:
            print(f"删除失败 - 文件夹: {folder_path_to_delete}, 错误: {e}")

    # 结果反馈
    print("\n操作完成:")
    print(f"  总病人数据: {len(subfolders)}")
    print(f"  保留文件: {latest_folder}")
    print(f"  删除数量: {deleted_count}")
    return True


def preprocess_folders(root_path):
    folders = [f for f in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, f))]
    for folder in folders:
        folder_path = os.path.join(root_path, folder)
        process_subfolders(folder_path)


class ProcessThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(set, int, int, int)
    error = pyqtSignal(str)

    def __init__(self, root_dir):
        super().__init__()
        self.root_dir = root_dir

    def run(self):
        try:
            all_labels = set()
            deleted_dates = 0
            deleted_patients = 0

            patient_dirs = [os.path.join(self.root_dir, d) for d in os.listdir(self.root_dir) if
                            os.path.isdir(os.path.join(self.root_dir, d))]
            total_patients = len(patient_dirs)
            self.progress.emit(0)

            for patient_idx, patient_dir in enumerate(patient_dirs):
                rtstruct_found = False
                latest_folder = None
                latest_time = None

                for entry in os.listdir(patient_dir):
                    entry_path = os.path.join(patient_dir, entry)
                    if os.path.isdir(entry_path):
                        entry_time = os.path.getmtime(entry_path)
                        if latest_time is None or entry_time > latest_time:
                            latest_time = entry_time
                            latest_folder = entry_path

                for entry in os.listdir(patient_dir):
                    entry_path = os.path.join(patient_dir, entry)
                    if os.path.isdir(entry_path) and entry_path != latest_folder:
                        shutil.rmtree(entry_path)
                        deleted_dates += 1
                if latest_folder:
                    for file in os.listdir(latest_folder):
                        if file.lower().startswith('rs') and file.endswith('.dcm'):
                            rtstruct_found = True
                            rtstruct_path = os.path.join(latest_folder, file)
                            rtss = pydicom.dcmread(rtstruct_path)
                            for roi in rtss.StructureSetROISequence:
                                all_labels.add(roi.ROIName.lower())
                            break

                if not rtstruct_found:
                    shutil.rmtree(patient_dir)
                    deleted_patients += 1
                self.progress.emit(int((patient_idx + 1) / total_patients * 100))

            remaining_patient_dirs = [d for d in os.listdir(self.root_dir) if
                                      os.path.isdir(os.path.join(self.root_dir, d))]
            remaining_patients = len(remaining_patient_dirs)

            self.finished.emit(all_labels, deleted_dates, deleted_patients, remaining_patients)

        except Exception as e:
            self.error.emit(str(e))


def recursive_listdir(path):
    """Recursively list all files in a directory."""
    files = []
    for entry in os.listdir(path):
        full_path = os.path.join(path, entry)
        if os.path.isdir(full_path):
            files.extend(recursive_listdir(full_path))
        else:
            files.append(full_path)
    return files


class DropArea(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)

        main_layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()

        self.path_label = QLabel("请选择或拖入文件夹")
        self.path_label.setAlignment(Qt.AlignCenter)
        self.path_label.setStyleSheet("""
            QLabel {
                border: 2px dashed #aaa;
                border-radius: 5px;
                padding: 20px;
                background: #f0f0f0;
            }
        """)

        self.select_btn = QPushButton("选择原始文件")
        self.process_btn = QPushButton("数据清洗")
        self.process_btn.setEnabled(False)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setFixedHeight(20)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #ccc;
                border-radius: 3px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
            }
        """)

        self.status_label = QLabel("")

        # 右侧标签
        self.checkbox_container = QWidget()
        self.checkbox_layout = QVBoxLayout()
        self.checkbox_container.setLayout(self.checkbox_layout)

        # 滚动区域
        scroll = QScrollArea()
        scroll.setWidget(self.checkbox_container)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        # 合并
        self.merge_btn = QPushButton("合并选中标签")
        self.merge_btn.setEnabled(False)
        self.merge_btn.clicked.connect(self.merge_labels)

        # 创建数据集按钮
        self.create_dataset_btn = QPushButton("创建数据集")
        self.create_dataset_btn.setEnabled(False)
        self.create_dataset_btn.clicked.connect(self.create_dataset)

        # 添加label_list
        self.label_list = QListWidget()
        self.label_list.setStyleSheet("""
            QListWidget {
                border: 1px solid #ccc;
                border-radius: 3px;
            }
        """)

        # 创建"创建训练目录"按钮
        self.create_training_dir_btn = QPushButton("创建训练目录", self)
        self.create_training_dir_btn.clicked.connect(self.create_training_directory)

        # 修改左侧布局，添加label_list
        left_layout.addWidget(self.path_label)
        left_layout.addWidget(self.select_btn)
        left_layout.addWidget(self.process_btn)
        left_layout.addWidget(self.progress_bar)
        left_layout.addWidget(self.status_label)
        left_layout.addWidget(QLabel("处理结果:"))
        left_layout.addWidget(self.label_list)  # 添加到左侧布局
        left_layout.addWidget(self.create_training_dir_btn)
        left_layout.addWidget(self.create_dataset_btn)
        left_layout.addWidget(self.merge_btn)
        left_layout.addStretch()

        right_layout.addWidget(QLabel("选择要合并的标签:"))
        right_layout.addWidget(scroll)

        # 设置左右布局的比例
        main_layout.addLayout(left_layout, 1)
        main_layout.addLayout(right_layout, 1)

        self.setLayout(main_layout)

        self.select_btn.clicked.connect(self.select_folder)
        self.process_btn.clicked.connect(self.process_data)

        self.folder_path = None
        self.process_thread = None
        # 存储复选框
        self.checkboxes = []

        # 实例变量记录文件夹路径
        self.nnUNet_raw_path = None
        self.nnUNet_preprocessed_path = None
        self.nnUNet_results_path = None

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            folder_path = urls[0].toLocalFile()
            if folder_path and folder_path.strip():
                self.set_folder_path(folder_path)

    def select_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "选择文件夹")
        if folder_path:
            self.set_folder_path(folder_path)

    def set_folder_path(self, path):
        self.folder_path = path
        self.path_label.setText(f"已选择: {path}")
        self.process_btn.setEnabled(True)

    def process_data(self):
        if not self.folder_path:
            return

        self.label_list.clear()
        for checkbox in self.checkboxes:
            self.checkbox_layout.removeWidget(checkbox)
            checkbox.deleteLater()
        self.checkboxes.clear()

        self.progress_bar.setVisible(True)
        self.process_btn.setEnabled(False)

        # 创建并启动处理线程
        self.process_thread = ProcessThread(self.folder_path)
        self.process_thread.progress.connect(self.update_progress)
        self.process_thread.finished.connect(self.on_process_finished)
        self.process_thread.error.connect(self.on_process_error)

        # 设置进度条
        patient_folders = [f for f in os.listdir(self.folder_path)
                           if os.path.isdir(os.path.join(self.folder_path, f))]
        self.progress_bar.setMaximum(len(patient_folders))

        # 启动线程
        self.process_thread.start()

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def update_label_checkboxes(self, labels):
        # 清除现有的复选框
        for checkbox in self.checkboxes:
            self.checkbox_layout.removeWidget(checkbox)
            checkbox.deleteLater()
        self.checkboxes.clear()

        # 创建新的复选框
        for label in labels:
            checkbox = QCheckBox(label)
            self.checkboxes.append(checkbox)
            self.checkbox_layout.addWidget(checkbox)

        self.merge_btn.setEnabled(True)
        self.create_dataset_btn.setEnabled(True)  # 同时启用创建数据集按钮

    def merge_labels(self):
        # 获取选中的标签
        selected_labels = [cb.text() for cb in self.checkboxes if cb.isChecked()]

        # 检查是否选择了至少两个标签
        if len(selected_labels) < 2:
            QMessageBox.warning(self, "警告", "至少选择两种标签用于合并")
            return

        # 获取新标签名
        new_label, ok = QInputDialog.getText(self, "合并标签",
                                             "请输入合并后的标签名称:",
                                             text="_".join(selected_labels))
        if not ok or not new_label:
            return

        # 添加处理进度条
        merge_progress = QProgressBar(self)
        merge_progress.setFixedHeight(20)  # 固定高度
        merge_progress.setStyleSheet("""
            QProgressBar {
                border: 1px solid #ccc;
                border-radius: 3px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
            }
        """)
        left_layout = self.layout().itemAt(0).layout()
        left_layout.insertWidget(left_layout.count() - 1, merge_progress)

        # 开始合并标签
        try:
            self.merge_btn.setEnabled(False)
            self.status_label.setText("正在合并标签...")
            self.status_label.setStyleSheet("color: #000;")

            # 获取总文件夹数
            patient_folders = [d for d in os.listdir(self.folder_path) if
                               os.path.isdir(os.path.join(self.folder_path, d))]
            total_folders = len(patient_folders)
            merge_progress.setMaximum(total_folders)

            processed_count = 0
            modified_count = 0

            # 遍历所有病人文件夹
            for patient_dir in patient_folders:
                processed_count += 1
                merge_progress.setValue(processed_count)
                self.status_label.setText(
                    f"正在处理: {patient_dir[:min(len(patient_dir), 10)]}... ({processed_count}/{total_folders})")
                QApplication.processEvents()  # 更新界面

                patient_path = os.path.join(self.folder_path, patient_dir)
                if not os.path.isdir(patient_path):
                    continue

                rtstruct_path = None
                for sub_dir in os.listdir(patient_path):
                    if rtstruct_path: break
                    sub_dir_path = os.path.join(patient_path, sub_dir)
                    if os.path.isfile(sub_dir_path):
                        continue
                    else:
                        plans_files = os.listdir(sub_dir_path)
                        for plans_file in plans_files:
                            if plans_file.lower().startswith('rs') and plans_file.endswith('.dcm'):
                                rtstruct_path = os.path.join(sub_dir_path, plans_file)
                                break

                if not rtstruct_path:
                    continue

                try:
                    ds = pydicom.dcmread(rtstruct_path)
                    roi_numbers_to_merge = []
                    for roi in ds.StructureSetROISequence:
                        if roi.ROIName.lower() in selected_labels:
                            roi_numbers_to_merge.append(roi.ROINumber)

                    if not roi_numbers_to_merge:
                        continue  # 跳过不合格用户

                    if len(roi_numbers_to_merge) == 1:
                        for roi in ds.StructureSetROISequence:
                            if roi.ROINumber == roi_numbers_to_merge[0]:
                                roi.ROIName = new_label
                                modified_count += 1
                                break
                    else:
                        new_contour_sequence = []
                        for contour in ds.ROIContourSequence:
                            if contour.ReferencedROINumber in roi_numbers_to_merge:
                                new_contour_sequence.extend(contour.ContourSequence)

                        first_roi_number = roi_numbers_to_merge[0]
                        for roi in ds.StructureSetROISequence:
                            if roi.ROINumber == first_roi_number:
                                roi.ROIName = new_label
                                break

                        ds.StructureSetROISequence = [
                            roi for roi in ds.StructureSetROISequence
                            if roi.ROINumber not in roi_numbers_to_merge[1:]
                        ]

                        ds.ROIContourSequence = [
                            contour for contour in ds.ROIContourSequence
                            if contour.ReferencedROINumber == first_roi_number
                        ]
                        ds.ROIContourSequence[0].ContourSequence = new_contour_sequence

                        modified_count += 1

                    ds.save_as(rtstruct_path)

                except Exception as e:
                    print(f"处理文件 {rtstruct_path} 时出错: {str(e)}")

            # 显示完成消息
            QMessageBox.information(self, "合并完成",
                                    f"标签合并完成！\n"
                                    f"- 处理了 {total_folders} 个病人文件夹\n"
                                    f"- 修改了 {modified_count} 个RTStructure文件")

            # 更新标签复选框
            all_labels = set()
            for patient_dir in os.listdir(self.folder_path):
                patient_path = os.path.join(self.folder_path, patient_dir)
                if not os.path.isdir(patient_path):
                    continue

                files = recursive_listdir(patient_path)
                for file in files:
                    if file.lower().startswith('rs') and file.endswith('.dcm'):
                        try:
                            ds = pydicom.dcmread(file)
                            for roi in ds.StructureSetROISequence:
                                all_labels.add(roi.ROIName.lower())
                        except:
                            continue

            # 更新标签复选框
            self.update_label_checkboxes(sorted(all_labels))

            self.status_label.setText("标签合并完成！")
            self.status_label.setStyleSheet("color: #4CAF50;")

        except Exception as e:
            self.status_label.setText(f"合并标签时出错: {str(e)}")
            self.status_label.setStyleSheet("color: #f44336;")
            QMessageBox.critical(self, "错误", f"合并标签时出错: {str(e)}")
        finally:
            self.merge_btn.setEnabled(True)
            # 移除进度条
            merge_progress.deleteLater()

    def rtstruct_to_combined_mask(self, rtss, image, roi_names):
        # 初始化掩码，与图像大小一致
        image_array = sitk.GetArrayFromImage(image)  # z, y, x
        combined_mask = np.zeros(image_array.shape, dtype=np.uint8)

        # 为每个 ROI 分配整数标签
        for roi_number, roi_name in roi_names.items():
            label_value = list(roi_names.values()).index(roi_name) + 1  # 背景为 0，ROI 从 1 开始

            # 找到对应的轮廓序列
            contour_sequence = [cs for cs in rtss.ROIContourSequence if cs.ReferencedROINumber == roi_number]
            if not contour_sequence:
                continue

            # 遍历每个轮廓
            for contour in contour_sequence[0].ContourSequence:
                points = np.array(contour.ContourData).reshape(-1, 3)
                # 转换为图像坐标
                indices = [image.TransformPhysicalPointToIndex(p) for p in points]
                x_coords = [idx[0] for idx in indices]  # x 坐标
                y_coords = [idx[1] for idx in indices]  # y 坐标
                z_slice = indices[0][2]  # z 坐标（假设同一轮廓在同一切片）

                # 检查边界
                if not all(
                        0 <= i < s for i, s in zip([min(x_coords), min(y_coords), z_slice], combined_mask.shape[::-1])):
                    continue

                # 填充多边形区域
                rr, cc = polygon(y_coords, x_coords, shape=combined_mask[z_slice].shape)
                combined_mask[z_slice, rr, cc] = label_value

        return combined_mask

    def create_dataset(self):
        # 获取选中的标签
        selected_labels = [cb.text() for cb in self.checkboxes if cb.isChecked()]

        # 检查是否选择了至少一个标签
        if not selected_labels:
            QMessageBox.warning(self, "警告", "请至少选择一个标签！")
            return

        # 获取数据集名称
        dataset_name, ok = QInputDialog.getText(self, "数据集名称", "请输入数据集名称:")
        if not ok or not dataset_name:
            return

        # 选择输出目录
        if self.nnUNet_raw_path is not None:
            output_dir = self.nnUNet_raw_path
        else:
            output_dir = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if not output_dir:
            return

        # 检查现有数据集并生成新的数据集名称
        existing_datasets = [d for d in os.listdir(output_dir) if d.startswith("Dataset")]
        max_index = 0
        for dataset in existing_datasets:
            parts = dataset.split('_')
            if len(parts) > 0 and parts[0][len("Dataset"):].isdigit():
                index = int(parts[0][len("Dataset"):])
                if index > max_index:
                    max_index = index

        self.dataset_index = max_index
        new_dataset_name = f"Dataset{max_index + 1:03d}_{dataset_name}"

        # 创建数据集文件夹
        dataset_folder = os.path.join(output_dir, new_dataset_name)
        image_folder = os.path.join(dataset_folder, "imagesTr")
        label_folder = os.path.join(dataset_folder, "labelsTr")
        os.makedirs(image_folder, exist_ok=True)
        os.makedirs(label_folder, exist_ok=True)

        # 添加处理进度条
        dataset_progress = QProgressBar(self)
        dataset_progress.setFixedHeight(20)  # 固定高度
        dataset_progress.setStyleSheet("""
            QProgressBar {
                border: 1px solid #ccc;
                border-radius: 3px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
            }
        """)
        left_layout = self.layout().itemAt(0).layout()
        left_layout.insertWidget(left_layout.count() - 1, dataset_progress)

        try:
            self.create_dataset_btn.setEnabled(False)
            self.status_label.setText("正在创建数据集...")
            self.status_label.setStyleSheet("color: #000;")

            # 获取所有病人文件夹
            patient_folders = [d for d in os.listdir(self.folder_path) if
                               os.path.isdir(os.path.join(self.folder_path, d))]
            total_folders = len(patient_folders)
            dataset_progress.setMaximum(total_folders)

            processed_count = 0
            valid_patients = 0
            deleted_patients = 0

            selected_labels_set = set(label.lower() for label in selected_labels)
            label_to_id = {label.lower(): idx + 1 for idx, label in enumerate(selected_labels)}

            # 初始化数据结构
            training_list = []

            # 遍历所有病人文件夹
            for patient_index, patient_dir in enumerate(patient_folders, start=1):
                processed_count += 1
                dataset_progress.setValue(processed_count)
                self.status_label.setText(f"正在处理: {patient_dir[:8]}... ({processed_count}/{total_folders})")
                QApplication.processEvents()

                patient_path = os.path.join(self.folder_path, patient_dir)
                if not os.path.isdir(patient_path):
                    continue

                # 初始化CT数据和RTStructure路径
                ct_series = []
                rtstruct_path = None

                # 遍历子目录以获取DICOM文件
                for sub_dir in os.listdir(patient_path):
                    sub_dir_path = os.path.join(patient_path, sub_dir)
                    if os.path.isfile(sub_dir_path) and sub_dir_path.endswith('.dcm'):
                        ct_series.append(sub_dir_path)
                    elif os.path.isdir(sub_dir_path):
                        for plans_file in os.listdir(sub_dir_path):
                            if plans_file.lower().startswith('rs') and plans_file.endswith('.dcm'):
                                rtstruct_path = os.path.join(sub_dir_path, plans_file)
                                break

                if not rtstruct_path:
                    print(f"未找到RTStructure文件，跳过病人: {patient_dir}")
                    continue

                ds = pydicom.dcmread(rtstruct_path)
                patient_labels = {roi.ROIName.lower() for roi in ds.StructureSetROISequence}
                # 检查是否包含所有选定的标签
                if not selected_labels_set.issubset(patient_labels):
                    print(f"病人 {patient_dir} 的RTStructure不包含所有选定标签，跳过该病人。")
                    continue
                else:
                    dicom_dir = os.path.join(patient_path, "dicom")
                    if not os.path.exists(dicom_dir):
                        os.makedirs(dicom_dir)
                    for ct in ct_series:
                        shutil.move(ct, dicom_dir)

                # 删除StructureSet中ROISequence中多余的ROI并根据label_to_id映射重新编号
                valid_roi_map = {
                    roi.ROIName.lower(): label_to_id[roi.ROIName.lower()]
                    for roi in ds.StructureSetROISequence
                    if roi.ROIName.lower() in selected_labels_set
                }

                # 过滤并更新StructureSetROISequence
                new_roi_sequence = []
                for roi in ds.StructureSetROISequence:
                    roi_name = roi.ROIName.lower()
                    if roi_name in valid_roi_map:
                        roi.ROINumber = valid_roi_map[roi_name]  # 直接修改属性
                        new_roi_sequence.append(roi)
                ds.StructureSetROISequence = new_roi_sequence

                # 过滤关联序列（使用集合查找优化性能）
                valid_numbers = set(valid_roi_map.values())
                ds.ROIContourSequence = [c for c in ds.ROIContourSequence if c.ReferencedROINumber in valid_numbers]
                ds.RTROIObservationsSequence = [o for o in ds.RTROIObservationsSequence if
                                                o.ReferencedROINumber in valid_numbers]
                ds.NumberOfROIContours = len(ds.ROIContourSequence)

                # 保存修改（保持原始传输语法）
                ds.save_as(rtstruct_path)

                # 读取DICOM序列并获取空间属性
                reader = sitk.ImageSeriesReader()
                dicom_files = reader.GetGDCMSeriesFileNames(dicom_dir)
                reader.SetFileNames(dicom_files)
                ct_volume = reader.Execute()
                # 保存CT图像

                ct_output_path = os.path.join(dataset_folder, "imagesTr", f"{valid_patients:03d}_0000.nii.gz")
                sitk.WriteImage(ct_volume, ct_output_path)

                # 1. 读取CT图像并获取空间属性
                ct_image = sitk.ReadImage(ct_output_path)
                spacing = ct_image.GetSpacing()  # 体素间距 (x,y,z)
                origin = ct_image.GetOrigin()  # 原点坐标
                direction = ct_image.GetDirection()  # 方向矩阵
                size = ct_image.GetSize()  # 图像维度 (w,h,d)

                # 2. 创建空白标签图像并转换为numpy数组
                mask = sitk.Image(size, sitk.sitkUInt8)
                mask.SetSpacing(spacing)
                mask.SetOrigin(origin)
                mask.SetDirection(direction)
                mask_array = sitk.GetArrayFromImage(mask)  # 初始化mask_array

                # 3. 读取RTSTRUCT文件
                rt_ds = pydicom.dcmread(rtstruct_path)
                roi_contour_sequence = rt_ds.ROIContourSequence

                # 4. 遍历所有ROI并绘制掩码
                for roi in roi_contour_sequence:
                    contour_sequence = roi.ContourSequence
                    for contour in contour_sequence:
                        try:
                            # 获取物理坐标点
                            points = np.array(contour.ContourData).reshape(-1, 3)

                            # 转换为体素坐标
                            voxel_coords = [
                                ct_image.TransformPhysicalPointToIndex(point.tolist())
                                for point in points
                            ]

                            # 提取切片索引并限制范围
                            z_index = int(np.clip(voxel_coords[0][2], 0, mask_array.shape[0] - 1))

                            # 生成二维多边形掩码
                            x_coords = [v[0] for v in voxel_coords]
                            y_coords = [v[1] for v in voxel_coords]
                            rr, cc = polygon(y_coords, x_coords)

                            # 更新mask_array
                            mask_array[z_index, rr, cc] = 1

                        except Exception as e:
                            print(f"绘制ROI失败: {str(e)}")
                            continue

                # 5. 将numpy数组转回SimpleITK图像
                mask = sitk.GetImageFromArray(mask_array)
                mask.CopyInformation(ct_image)  # 继承空间属性

                # 6. 保存掩码文件
                label_output_path = os.path.join(label_folder, f"{valid_patients:03d}.nii.gz")
                sitk.WriteImage(mask, label_output_path)

                # 更新训练列表
                training_list.append({
                    "image": f"./imagesTr/{valid_patients}_0000.nii.gz",
                    "label": f"./labelsTr/{valid_patients}.nii.gz"
                })
                valid_patients += 1

            # 生成dataset.json
            dataset_json = {
                "name": dataset_name,
                "description": f"Automatic segmentation of RT structures ({datetime.now().strftime('%Y-%m')})",
                "tensorImageSize": "3D",
                "channel_names": {"0": "CT"},
                "labels": {"background": 0, **{label: int(i)+1 for i, label in enumerate(selected_labels)}},
                "numTraining": len(training_list),
                "training": training_list,
                "file_ending": ".nii.gz"
            }

            with open(os.path.join(dataset_folder, "dataset.json"), 'w') as f:
                json.dump(dataset_json, f, indent=4)

            # 显示完成消息
            QMessageBox.information(self, "数据集创建完成",
                                    f"数据集创建完成！\n"
                                    f"- 保留了 {valid_patients} 个完整数据的病人\n"
                                    f"- 删除了 {deleted_patients} 个不完整数据的病人\n"
                                    f"- 选定的标签集合: {', '.join(selected_labels)}")

            self.status_label.setText("数据集创建完成！")
            self.status_label.setStyleSheet("color: #4CAF50;")

        except Exception as e:
            self.status_label.setText(f"创建数据集时出错: {str(e)}")
            self.status_label.setStyleSheet("color: #f44336;")
            QMessageBox.critical(self, "错误", f"创建数据集时出错: {str(e)}")
        finally:
            self.create_dataset_btn.setEnabled(True)
            # 移除进度条
            dataset_progress.deleteLater()

    def on_process_finished(self, sorted_labels, deleted_dates, deleted_patients, remaining_patients):
        # 更新状态标签
        self.status_label.setText(
            f"处理完成:\n"
            f"- 删除了 {deleted_dates} 个重复日期文件夹\n"
            f"- 删除了 {deleted_patients} 个缺少RTStructure的用户文件夹\n"
            f"- 当前阶段的病人数据数目: {remaining_patients}"
        )
        self.status_label.setStyleSheet("color: #4CAF50;")
        self.progress_bar.setVisible(False)
        self.process_btn.setEnabled(True)

        # 更新标签复选框
        self.update_label_checkboxes(sorted_labels)

    def on_process_error(self, error_msg):
        self.status_label.setText(f"处理出错: {error_msg}")
        self.status_label.setStyleSheet("color: #f44336;")
        self.progress_bar.setVisible(False)
        self.process_btn.setEnabled(True)

    def create_training_directory(self):
        # 选择目录
        training_dir = QFileDialog.getExistingDirectory(self, "选择训练目录")
        if training_dir:
            # 创建子目录
            self.nnUNet_raw_path = os.path.join(training_dir, "nnUNet_raw")
            self.nnUNet_preprocessed_path = os.path.join(training_dir, "nnUNet_preprocessed")
            self.nnUNet_results_path = os.path.join(training_dir, "nnUNet_results")

            os.makedirs(self.nnUNet_raw_path, exist_ok=True)
            os.makedirs(self.nnUNet_preprocessed_path, exist_ok=True)
            os.makedirs(self.nnUNet_results_path, exist_ok=True)

            QMessageBox.information(self, "成功", "训练目录已创建！")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("医学影像数据预处理工具")
        self.setMinimumSize(800, 600)

        # 创建中心部件
        central_widget = DropArea()
        self.setCentralWidget(central_widget)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
