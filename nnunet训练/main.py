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


def get_RTStructure_label(root_dir):
    all_labels = set()
    for patient_idx, patient_dir in enumerate(os.listdir(root_dir)):
        patient_path = os.path.join(root_dir, patient_dir)
        meta_file = None

        for sub_dir in os.listdir(patient_path):
            sub_dir_path = os.path.join(patient_path, sub_dir)
            if os.path.isfile(sub_dir_path) and sub_dir_path.endswith('.dcm'):
                continue
            else:
                plnas_files = os.listdir(sub_dir_path)
                for plnas_file in plnas_files:
                    if plnas_file.lower().startswith('rs') and plnas_file.endswith('.dcm'):
                        meta_file = os.path.join(sub_dir_path, plnas_file)
                        break
                if meta_file is None:
                    print("缺少RTStructure文件,跳过该用户")
                    break

        if meta_file:
            rtss = pydicom.dcmread(meta_file)
            label_mapping = {}

            for roi in rtss.StructureSetROISequence:
                roi_number = roi.ROINumber
                roi_name = roi.ROIName.lower()
                label_mapping[roi_number] = roi_name
                all_labels.add(roi_name)
            print(patient_idx + 1, "=====", sorted(label_mapping.items()))
    return all_labels


class ProcessThread(QThread):
    progress = pyqtSignal(int)  # 进度信号
    finished = pyqtSignal(list, int, int)  # 完成信号：标签列表, 删除的重复日期数, 删除的无效用户数
    error = pyqtSignal(str)  # 错误信号

    def __init__(self, folder_path):
        super().__init__()
        self.folder_path = folder_path

    def run(self):
        try:
            deleted_dates = 0  # 统计删除的重复日期文件夹数
            deleted_patients = 0  # 统计删除的无效用户数

            # 处理多余文件夹并统计删除的文件夹数
            patient_folders = [f for f in os.listdir(self.folder_path)
                               if os.path.isdir(os.path.join(self.folder_path, f))]

            # 第一遍遍历：处理日期文件夹
            for folder in patient_folders:
                folder_path = os.path.join(self.folder_path, folder)
                subfolders = [f for f in os.listdir(folder_path)
                              if os.path.isdir(os.path.join(folder_path, f))]
                if len(subfolders) <= 1:
                    continue

                date_pattern = re.compile(r'\d{4}[-_]?\d{2}[-_]?\d{2}')
                folder_dates = {}
                for subfolder in subfolders:
                    match = date_pattern.search(subfolder)
                    if match:
                        date_str = match.group().replace('-', '').replace('_', '')
                        try:
                            folder_date = datetime.strptime(date_str, '%Y%m%d')
                            folder_dates[subfolder] = folder_date
                        except ValueError:
                            continue

                if folder_dates:
                    sorted_folders = sorted(folder_dates.items(), key=lambda x: x[1], reverse=True)
                    latest_folder = sorted_folders[0][0]

                    # 统计删除的文件夹数
                    for subfolder in subfolders:
                        if subfolder != latest_folder:
                            deleted_dates += 1
                            folder_to_delete = os.path.join(folder_path, subfolder)
                            try:
                                import shutil
                                shutil.rmtree(folder_to_delete)
                            except Exception as e:
                                print(f"Error deleting folder {folder_to_delete}: {e}")

            # 第二遍遍历：检查并处理缺少RTStructure的用户
            patient_folders = [f for f in os.listdir(self.folder_path)
                               if os.path.isdir(os.path.join(self.folder_path, f))]
            all_labels = set()

            for idx, patient_dir in enumerate(patient_folders):
                self.progress.emit(idx + 1)
                patient_path = os.path.join(self.folder_path, patient_dir)
                has_rtstructure = False

                # 查找RTStructure文件
                for root, _, files in os.walk(patient_path):
                    for file in files:
                        if file.lower().startswith('rs') and file.endswith('.dcm'):
                            try:
                                rtss = pydicom.dcmread(os.path.join(root, file))
                                for roi in rtss.StructureSetROISequence:
                                    roi_name = roi.ROIName.lower()
                                    all_labels.add(roi_name)
                                has_rtstructure = True
                                break
                            except:
                                continue
                    if has_rtstructure:
                        break

                # 如果没有找到有效的RTStructure文件，删除该用户文件夹
                if not has_rtstructure:
                    try:
                        import shutil
                        shutil.rmtree(patient_path)
                        deleted_patients += 1
                        print(f"删除缺少RTStructure的用户: {patient_dir}")
                    except Exception as e:
                        print(f"删除用户文件夹失败 {patient_dir}: {e}")

            # 发送结果，包括统计信息
            self.finished.emit(sorted(all_labels), deleted_dates, deleted_patients)

        except Exception as e:
            self.error.emit(str(e))


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
        if len(selected_labels) < 2:
            self.status_label.setText("请至少选择两个标签进行合并！")
            self.status_label.setStyleSheet("color: #f44336;")
            return

        # 获取新标签名
        new_label, ok = QInputDialog.getText(self, "合并标签",
                                             "请输入合并后的标签名称:",
                                             text="_".join(selected_labels))
        if not ok or not new_label:
            return

        # 添加处理进度条
        merge_progress = QProgressBar(self)
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
            patient_folders = [d for d in os.listdir(self.folder_path) if os.path.isdir(os.path.join(self.folder_path, d))]
            total_folders = len(patient_folders)
            merge_progress.setMaximum(total_folders)

            processed_count = 0
            modified_count = 0

            # 遍历所有病人文件夹
            for patient_dir in patient_folders:
                processed_count += 1
                merge_progress.setValue(processed_count)
                self.status_label.setText(f"正在处理: {patient_dir} ({processed_count}/{total_folders})")
                QApplication.processEvents()  # 更新界面

                patient_path = os.path.join(self.folder_path, patient_dir)
                if not os.path.isdir(patient_path):
                    continue

                # 查找RTStructure文件
                for root, _, files in os.walk(patient_path):
                    for file in files:
                        if file.lower().startswith('rs') and file.endswith('.dcm'):
                            file_path = os.path.join(root, file)
                            try:
                                # 读取DICOM文件
                                ds = pydicom.dcmread(file_path)
                                modified = False

                                # 创建ROI名称到编号的映射
                                roi_name_to_number = {}
                                for roi in ds.StructureSetROISequence:
                                    roi_name_to_number[roi.ROIName.lower()] = roi.ROINumber

                                # 找到要合并的ROI的编号
                                roi_numbers_to_merge = []
                                for label in selected_labels:
                                    if label.lower() in roi_name_to_number:
                                        roi_numbers_to_merge.append(roi_name_to_number[label.lower()])

                                if roi_numbers_to_merge:
                                    # 更新第一个ROI的名称
                                    first_roi_number = roi_numbers_to_merge[0]
                                    for roi in ds.StructureSetROISequence:
                                        if roi.ROINumber == first_roi_number:
                                            roi.ROIName = new_label
                                            modified = True

                                    # 删除其他ROI
                                    if len(roi_numbers_to_merge) > 1:
                                        ds.StructureSetROISequence = [
                                            roi for roi in ds.StructureSetROISequence
                                            if roi.ROINumber not in roi_numbers_to_merge[1:]
                                        ]
                                        modified = True

                                if modified:
                                    ds.save_as(file_path)
                                    modified_count += 1

                            except Exception as e:
                                print(f"处理文件 {file_path} 时出错: {str(e)}")

            # 处理完成后更新标签列表
            all_labels = set()
            for patient_dir in os.listdir(self.folder_path):
                patient_path = os.path.join(self.folder_path, patient_dir)
                if not os.path.isdir(patient_path):
                    continue

                for root, _, files in os.walk(patient_path):
                    for file in files:
                        if file.lower().startswith('rs') and file.endswith('.dcm'):
                            try:
                                ds = pydicom.dcmread(os.path.join(root, file))
                                for roi in ds.StructureSetROISequence:
                                    all_labels.add(roi.ROIName.lower())
                            except:
                                continue

            # 显示完成消息
            QMessageBox.information(self, "合并完成",
                                    f"标签合并完成！\n"
                                    f"- 处理了 {total_folders} 个病人文件夹\n"
                                    f"- 修改了 {modified_count} 个RTStructure文件")

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

    def convert_dicom_to_nnunet(self, root_dir, output_dir, dataset_id=1, dataset_name="RT_Structures"):
        """
        参数说明：
        root_dir：原始DICOM数据根目录（包含各个患者子目录）
        output_dir：输出目录（会自动创建nnUNet要求的文件夹结构）
        dataset_id：数据集ID（三位整数，如1）
        dataset_name：数据集名称（如Liver_Tumors）
        """
        # 创建nnUNet标准目录结构
        dataset_folder = f"Dataset{dataset_id:03d}_{dataset_name}"
        output_path = os.path.join(output_dir, dataset_folder)
        os.makedirs(os.path.join(output_path, "imagesTr"), exist_ok=True)
        os.makedirs(os.path.join(output_path, "labelsTr"), exist_ok=True)

        # 初始化数据结构
        all_modalities = {"0": "CT"}  # 模态信息
        label_mapping = OrderedDict({"0": "background"})
        training_list = []

        # 遍历患者目录
        patient_dirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if
                        os.path.isdir(os.path.join(root_dir, d))]
        for patient_id, patient_dir in enumerate(patient_dirs):
            # 查找CT序列和RTSTRUCT文件
            ct_series = []
            rtstruct_path = None

            for sub_dir in os.listdir(patient_dir):
                sub_dir_path = os.path.join(patient_dir, sub_dir)
                if os.path.isfile(sub_dir_path) and sub_dir_path.endswith('.dcm'):
                    ct_series.append(sub_dir_path)
                else:
                    plnas_files = os.listdir(sub_dir_path)
                    for plans_file in plnas_files:
                        if plans_file.lower().startswith('rs') and plans_file.endswith('.dcm'):
                            rtstruct_path = os.path.join(sub_dir_path, plans_file)
                            break

            # 转换CT图像为NIfTI
            ct_series.sort(key=lambda x: pydicom.dcmread(x).InstanceNumber)
            reader = sitk.ImageSeriesReader()
            reader.SetFileNames(ct_series)
            ct_volume = reader.Execute()

            # 保存CT图像
            ct_output_path = os.path.join(output_path, "imagesTr", f"{patient_id}_0000.nii.gz")
            sitk.WriteImage(ct_volume, ct_output_path)

            # 处理RTSTRUCT文件
            rtss = pydicom.dcmread(rtstruct_path)
            roi_names = {}
            for roi in rtss.StructureSetROISequence:
                roi_number = roi.ROINumber
                roi_name = roi.ROIName.strip().lower()
                roi_names[roi_number] = roi_name
                if roi_name not in label_mapping.values():
                    label_mapping[str(len(label_mapping))] = roi_name

            # 创建标签图像
            reference_image = sitk.ReadImage(ct_series[0])
            label_array = np.zeros(sitk.GetArrayFromImage(ct_volume).shape, dtype=np.uint8)

            # 解析ROI轮廓数据
            for contour in rtss.ROIContourSequence:
                roi_number = contour.ReferencedROINumber
                if roi_number not in roi_names:
                    continue

                # 获取当前ROI的标签值
                label_value = [k for k, v in label_mapping.items() if v == roi_names[roi_number]][0]

                # 处理每个轮廓序列
                if hasattr(contour, 'ContourSequence'):
                    for contour_seq in contour.ContourSequence:
                        # 转换轮廓坐标到体素空间
                        contour_data = np.array(contour_seq.ContourData).reshape(-1, 3)
                        contour_indices = [ct_volume.TransformPhysicalPointToIndex(point)
                                           for point in contour_data]

                        # 使用SimpleITK绘制多边形填充
                        for idx in contour_indices:
                            try:
                                label_array[idx[2], idx[1], idx[0]] = int(label_value)
                            except IndexError:
                                continue

            # 保存标签图像
            label_image = sitk.GetImageFromArray(label_array)
            label_image.CopyInformation(ct_volume)
            label_output_path = os.path.join(output_path, "labelsTr", f"{patient_id}.nii.gz")
            sitk.WriteImage(label_image, label_output_path)

            # 添加到训练列表
            training_list.append({
                "image": f"./imagesTr/{patient_id}_0000.nii.gz",
                "label": f"./labelsTr/{patient_id}.nii.gz"
            })

        # 生成dataset.json
        dataset_json = OrderedDict({
            "name": dataset_name,
            "description": f"Automatic segmentation of RT structures ({datetime.now().strftime('%Y-%m')})",
            "tensorImageSize": "3D",
            "modality": all_modalities,
            "labels": label_mapping,
            "numTraining": len(training_list),
            "training": training_list,
            "file_ending": ".nii.gz"
        })

        with open(os.path.join(output_path, "dataset.json"), 'w') as f:
            json.dump(dataset_json, f, indent=4)

        print(f"Conversion complete. Dataset saved to: {output_path}")

    def create_dataset(self):
        # 获取选中的标签
        selected_labels = [cb.text() for cb in self.checkboxes if cb.isChecked()]
        if not selected_labels:
            self.status_label.setText("请至少选择一个标签！")
            self.status_label.setStyleSheet("color: #f44336;")
            return

        # 获取数据集名称
        dataset_name, ok = QInputDialog.getText(self, "数据集名称", "请输入数据集名称:")
        if not ok or not dataset_name:
            return

        # 选择输出目录
        output_dir = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if not output_dir:
            return

        # 添加处理进度条
        dataset_progress = QProgressBar(self)
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

            # 遍历所有病人文件夹
            for patient_dir in patient_folders:
                processed_count += 1
                dataset_progress.setValue(processed_count)
                self.status_label.setText(f"正在处理: {patient_dir} ({processed_count}/{total_folders})")
                QApplication.processEvents()

                patient_path = os.path.join(self.folder_path, patient_dir)
                if not os.path.isdir(patient_path):
                    continue

                # 查找RTStructure文件
                for root, _, files in os.walk(patient_path):
                    for file in files:
                        if file.lower().startswith('rs') and file.endswith('.dcm'):
                            file_path = os.path.join(root, file)
                            try:
                                # 读取DICOM文件
                                ds = pydicom.dcmread(file_path)

                                # 获取当前病人的所有标签
                                patient_labels = set()
                                for roi in ds.StructureSetROISequence:
                                    patient_labels.add(roi.ROIName.lower())

                                # 检查是否包含所有选定的标签
                                if selected_labels_set.issubset(patient_labels):
                                    # 保留选定的标签，删除其他标签
                                    new_structure_set = []
                                    for roi in ds.StructureSetROISequence:
                                        roi_name = roi.ROIName.lower()
                                        if roi_name in selected_labels_set:
                                            roi.ROINumber = label_to_id[roi_name]
                                            new_structure_set.append(roi)

                                    # 按照ROINumber排序
                                    new_structure_set.sort(key=lambda x: x.ROINumber)
                                    ds.StructureSetROISequence = new_structure_set

                                    # 更新关联序列
                                    for seq in getattr(ds, 'RTROIObservationsSequence', []):
                                        if hasattr(seq, 'ReferencedROINumber'):
                                            seq.ReferencedROINumber = label_to_id.get(seq.ROIName.lower(),
                                                                                      seq.ReferencedROINumber)

                                    if hasattr(ds, 'ROIContourSequence'):
                                        for cs in ds.ROIContourSequence:
                                            cs.ReferencedROINumber = label_to_id.get(cs.ROIName.lower(),
                                                                                     cs.ReferencedROINumber)

                                    ds.save_as(file_path)
                                    valid_patients += 1
                                else:
                                    # 删除不符合要求的病人文件夹
                                    import shutil
                                    shutil.rmtree(patient_path)
                                    deleted_patients += 1
                                    print(f"删除不完整数据的用户: {patient_dir}")
                                break
                            except Exception as e:
                                print(f"处理文件 {file_path} 时出错: {str(e)}")
                            break
                    break

            # 调用convert_dicom_to_nnunet函数
            self.convert_dicom_to_nnunet(self.folder_path, output_dir, dataset_id=1, dataset_name=dataset_name)

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

    def on_process_finished(self, sorted_labels, deleted_dates, deleted_patients):
        # 不再需要添加到label_list，直接更新复选框
        self.status_label.setText(
            f"处理完成:\n"
            f"- 删除了 {deleted_dates} 个重复日期文件夹\n"
            f"- 删除了 {deleted_patients} 个缺少RTStructure的用户文件夹"
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
