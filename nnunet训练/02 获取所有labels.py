import os
import SimpleITK as sitk
import numpy as np
import pandas as pd
import pydicom
import shutil

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
            print(patient_idx+1, "=====",sorted(label_mapping.items()))
    return all_labels


def alighment_RTStructure(root_dir, labels_dict):
    ERROR_LABEL_MAPPING = {"sigmiod": "sigmoid"}

    for patient_dir in os.listdir(root_dir):
        patient_path = os.path.join(root_dir, patient_dir)
        meta_file = None

        for root, _, files in os.walk(patient_path):
            for file in files:
                if file.lower().startswith('rs') and file.lower().endswith('.dcm'):
                    meta_file = os.path.join(root, file)
                    break
            if meta_file:
                break

        if not meta_file:
            print(f"× 跳过患者 {patient_dir}: 未找到RTStructure文件")
            continue

        # 2. 安全备份
        backup_path = meta_file + ".bak"
        try:
            if not os.path.exists(backup_path):
                shutil.copy2(meta_file, backup_path)  # 保留元数据的完整备份
        except PermissionError:
            print(f"! 权限拒绝: 无法创建备份文件 {backup_path}")
            continue

        # 3. 读取并修改DICOM文件
        try:
            rtss = pydicom.dcmread(meta_file)
            original_roi_mapping = {}

            # 3.1 标签修正与编号映射
            for roi in rtss.StructureSetROISequence:
                # 标签名称规范化处理
                raw_name = roi.ROIName.strip().lower()
                corrected_name = ERROR_LABEL_MAPPING.get(raw_name, raw_name)

                # 名称修正逻辑
                if corrected_name != raw_name:
                    print(f"! 修正标签: {patient_dir} [{raw_name} → {corrected_name}]")
                    roi.ROIName = corrected_name  # 更新为正确名称

                # 编号对齐检查
                if corrected_name not in labels_dict:
                    raise ValueError(f"患者 {patient_dir} 包含非法标签：{corrected_name}（全局字典中不存在）")

                # 记录编号映射关系
                original_roi_mapping[roi.ROINumber] = labels_dict[corrected_name]
                roi.ROINumber = labels_dict[corrected_name]

            # 3.2 更新关联序列（增强兼容性）
            REFERENCED_FIELDS = ['ReferencedROINumber', 'RefdROINumber']
            for seq in getattr(rtss, 'RTROIObservationsSequence', []):
                for field in REFERENCED_FIELDS:
                    if hasattr(seq, field):
                        old_num = getattr(seq, field)
                        setattr(seq, field, original_roi_mapping.get(old_num, old_num))

            # 补充对 ROIContourSequence 的引用更新
            if hasattr(rtss, 'ROIContourSequence'):
                for cs in rtss.ROIContourSequence:
                    old_num = cs.ReferencedROINumber
                    if old_num in original_roi_mapping:
                        cs.ReferencedROINumber = original_roi_mapping[old_num]
                    else:
                        raise ValueError(f"患者 {patient_dir} 中未找到 ROI 编号 {old_num} 的映射关系")

            # 4. 强制刷新DICOM元数据
            rtss.file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()  # 新UID
            rtss.SOPInstanceUID = rtss.file_meta.MediaStorageSOPInstanceUID
            rtss.InstitutionName = "Aligned by SDC System"  # 添加修改标识

            # 5. 安全写入检查
            if not os.access(meta_file, os.W_OK):
                raise PermissionError(f"禁止写入: {meta_file}")

            temp_file = meta_file + ".tmp"
            rtss.save_as(temp_file)
            shutil.move(temp_file, meta_file)
            print(f"√ 成功更新: {patient_dir} → {meta_file}")

        except Exception as e:
            print(f"! 严重错误: {patient_dir} - {str(e)}")
            if os.path.exists(backup_path):
                shutil.move(backup_path, meta_file)  # 自动恢复备份
                print(f"! 已恢复备份: {meta_file}")


if __name__ == "__main__":
    labels = get_RTStructure_label(r"D:\DevData\SDC003_200")
    # 去除错误的sigmiod标签
    if "sigmiod" in labels:
        labels.remove("sigmiod")
    # 按照字典序排序
    labels = sorted(labels)
    # 创建字典
    labels_dict = dict(zip(labels, range(1, len(labels) + 1)))
    print(labels_dict)
    # 重新遍历调整rtstructure文件
    # alighment_RTStructure(r"D:\DevData\SDC003_200", labels_dict=labels_dict)
