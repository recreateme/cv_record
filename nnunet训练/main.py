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
import os
import re
import json
from collections import OrderedDict
import SimpleITK as sitk
import numpy as np

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

    if len(subfolders) < 1:
        print("无需处理,子文件夹数量≤1")
        return False
    date_pattern = re.compile(r'(\d{4})[-_]?(\d{2})[-_]?(\d{2})')
    valid_date_folders = {}

    for folder in subfolders:
        match = date_pattern.search(folder)
        if match:
            try:
                # 拼接为YYYYMMDD格式并解析
                year, month, day = match.groups()
                date_str = f"{year}{month}{day}"
                folder_date = datetime.datetime.strptime(date_str, "%Y%m%d")
                valid_date_folders[folder] = folder_date
            except (ValueError, TypeError) as e:
                print(f"日期解析失败 - 文件夹: {folder}, 错误: {e}")

    # 文件夹排序
    latest_folder = None
    if valid_date_folders:
        # 按日期降序排序
        sorted_dates = sorted(valid_date_folders.items(), key=lambda x: x[1], reverse=True)
        latest_folder = sorted_dates[0][0]
        print(f"检测到标准日期格式,保留最新: {latest_folder}")
    else:
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

                # 遍历用户目录下的所有文件和文件夹
                for entry in os.listdir(patient_dir):
                    entry_path = os.path.join(patient_dir, entry)
                    if os.path.isdir(entry_path):
                        # 检查文件夹的修改时间
                        entry_time = os.path.getmtime(entry_path)
                        if latest_time is None or entry_time > latest_time:
                            latest_time = entry_time
                            latest_folder = entry_path

                # 删除多余的文件夹
                for entry in os.listdir(patient_dir):
                    entry_path = os.path.join(patient_dir, entry)
                    if os.path.isdir(entry_path) and entry_path != latest_folder:
                        shutil.rmtree(entry_path)
                        deleted_dates += 1

                # 检查最新文件夹中是否有RTStructure文件
                if latest_folder:
                    for file in os.listdir(latest_folder):
                        if file.lower().startswith('rs') and file.endswith('.dcm'):
                            rtstruct_found = True
                            rtstruct_path = os.path.join(latest_folder, file)
                            # 读取RTStructure文件
                            rtss = pydicom.dcmread(rtstruct_path)
                            for roi in rtss.StructureSetROISequence:
                                all_labels.add(roi.ROIName.lower())
                            break

                # 如果没有找到RTStructure文件，删除整个用户目录
                if not rtstruct_found:
                    shutil.rmtree(patient_dir)
                    deleted_patients += 1

                # 更新进度条
                self.progress.emit(int((patient_idx + 1) / total_patients * 100))

            # 计算处理后的病人数据数目
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

        # 创建主布局
        main_layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()

        # 左侧控件
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

        self.select_btn = QPushButton("选择文件夹")
        self.process_btn = QPushButton("处理文件夹")
        self.process_btn.setEnabled(False)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setFixedHeight(20)  # 固定高度
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

        # 右侧标签选择区域
        self.checkbox_container = QWidget()
        self.checkbox_layout = QVBoxLayout()
        self.checkbox_container.setLayout(self.checkbox_layout)

        # 创建滚动区域
        scroll = QScrollArea()
        scroll.setWidget(self.checkbox_container)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        # 合并按钮
        self.merge_btn = QPushButton("合并选中标签")
        self.merge_btn.setEnabled(False)
        self.merge_btn.clicked.connect(self.merge_labels)

        # 创建数据集按钮
        self.create_dataset_btn = QPushButton("创建数据集")
        self.create_dataset_btn.setEnabled(False)
        self.create_dataset_btn.clicked.connect(self.create_dataset)

        # 添加label_list用于显示处理结果
        self.label_list = QListWidget()
        self.label_list.setStyleSheet("""
            QListWidget {
                border: 1px solid #ccc;
                border-radius: 3px;
            }
        """)

        # 修改左侧布局，添加label_list
        left_layout.addWidget(self.path_label)
        left_layout.addWidget(self.select_btn)
        left_layout.addWidget(self.process_btn)
        left_layout.addWidget(self.progress_bar)
        left_layout.addWidget(self.status_label)
        left_layout.addWidget(QLabel("处理结果:"))
        left_layout.addWidget(self.label_list)  # 添加到左侧布局
        left_layout.addStretch()

        right_layout.addWidget(QLabel("选择要合并的标签:"))
        right_layout.addWidget(scroll)
        right_layout.addWidget(self.merge_btn)
        right_layout.addWidget(QLabel("选择要创建数据集的标签:"))
        right_layout.addWidget(self.create_dataset_btn)

        # 设置左右布局的比例
        main_layout.addLayout(left_layout, 1)
        main_layout.addLayout(right_layout, 1)

        self.setLayout(main_layout)

        # 连接信号
        self.select_btn.clicked.connect(self.select_folder)
        self.process_btn.clicked.connect(self.process_data)

        self.folder_path = None
        self.process_thread = None
        self.checkboxes = []  # 存储复选框

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

        # 清空列表和复选框
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

                # 查找RTStructure文件
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

                # 读取RTStructure文件
                try:
                    ds = pydicom.dcmread(rtstruct_path)
                    roi_numbers_to_merge = []
                    for roi in ds.StructureSetROISequence:
                        if roi.ROIName.lower() in selected_labels:
                            roi_numbers_to_merge.append(roi.ROINumber)

                    if not roi_numbers_to_merge:
                        continue  # 跳过不包含选中标签的用户

                    if len(roi_numbers_to_merge) == 1:
                        # 仅有一个标签，直接修改RTStructure文件
                        for roi in ds.StructureSetROISequence:
                            if roi.ROINumber == roi_numbers_to_merge[0]:
                                roi.ROIName = new_label
                                modified_count += 1
                                break
                    else:
                        # 多个标签，合并轮廓信息
                        new_contour_sequence = []
                        for contour in ds.ROIContourSequence:
                            if contour.ReferencedROINumber in roi_numbers_to_merge:
                                new_contour_sequence.extend(contour.ContourSequence)

                        # 更新第一个ROI的名称
                        first_roi_number = roi_numbers_to_merge[0]
                        for roi in ds.StructureSetROISequence:
                            if roi.ROINumber == first_roi_number:
                                roi.ROIName = new_label
                                break

                        # 删除其他ROI
                        ds.StructureSetROISequence = [
                            roi for roi in ds.StructureSetROISequence
                            if roi.ROINumber not in roi_numbers_to_merge[1:]
                        ]

                        # 更新轮廓序列
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

        new_dataset_name = f"Dataset{max_index + 1:03d}_{dataset_name}"

        # 创建数据集文件夹
        dataset_folder = os.path.join(output_dir, new_dataset_name)
        os.makedirs(os.path.join(dataset_folder, "imagesTr"), exist_ok=True)
        os.makedirs(os.path.join(dataset_folder, "labelsTr"), exist_ok=True)

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
            patient_folders = [d for d in os.listdir(self.folder_path)
                               if os.path.isdir(os.path.join(self.folder_path, d))]
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
            for patient_dir in patient_folders:
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
                    continue

                # 读取RTStructure文件
                try:
                    ds = pydicom.dcmread(rtstruct_path)
                    patient_labels = {roi.ROIName.lower() for roi in ds.StructureSetROISequence}

                    # 检查是否包含所有选定的标签
                    if not selected_labels_set.issubset(patient_labels):
                        continue

                    # 删除多余标签的轮廓信息
                    new_structure_set = []
                    for roi in ds.StructureSetROISequence:
                        roi_name = roi.ROIName.lower()
                        if roi_name in selected_labels_set:
                            new_structure_set.append(roi)

                    ds.StructureSetROISequence = new_structure_set

                    # 保存RTStructure为NIfTI格式
                    rtstruct_output_path = os.path.join(dataset_folder, "labelsTr", f"{patient_dir}.nii.gz")
                    ds.save_as(rtstruct_output_path)

                    # 转换CT图像为NIfTI格式
                    ct_series.sort(key=lambda x: pydicom.dcmread(x).InstanceNumber)
                    reader = sitk.ImageSeriesReader()
                    reader.SetFileNames(ct_series)
                    ct_volume = reader.Execute()

                    # 保存CT图像
                    ct_output_path = os.path.join(dataset_folder, "imagesTr", f"{patient_dir}_0000.nii.gz")
                    sitk.WriteImage(ct_volume, ct_output_path)

                    # 添加到训练列表
                    training_list.append({
                        "image": f"./imagesTr/{patient_dir}_0000.nii.gz",
                        "label": f"./labelsTr/{patient_dir}.nii.gz"
                    })

                    valid_patients += 1

                except Exception as e:
                    print(f"处理文件 {rtstruct_path} 时出错: {str(e)}")
                    deleted_patients += 1

            # 生成dataset.json
            dataset_json = {
                "name": dataset_name,
                "description": f"Automatic segmentation of RT structures ({datetime.now().strftime('%Y-%m')})",
                "tensorImageSize": "3D",
                "channel_names": {"0": "CT"},
                "labels": {"background": 0, **{label: int(i) for i, label in enumerate(selected_labels)}},
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
