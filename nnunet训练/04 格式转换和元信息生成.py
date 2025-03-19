import os
import pydicom
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
import json
from datetime import datetime
from collections import OrderedDict


def convert_dicom_to_nnunet(root_dir, output_dir, dataset_id=1, dataset_name="RT_Structures"):
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
    patient_dirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    print(patient_dirs)
    for patient_id, patient_dir in enumerate(patient_dirs):
        # 查找CT序列和RTSTRUCT文件[2](@ref)
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

        # 转换CT图像为NIfTI[2](@ref)
        # 按实例编号排序CT序列
        ct_series.sort(key=lambda x: pydicom.dcmread(x).InstanceNumber)
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(ct_series)
        ct_volume = reader.Execute()

        # 保存CT图像
        ct_output_path = os.path.join(output_path, "imagesTr", f"{patient_id}_0000.nii.gz")
        sitk.WriteImage(ct_volume, ct_output_path)

        # 处理RTSTRUCT文件[4](@ref)
        rtss = pydicom.dcmread(rtstruct_path)
        roi_names = {}
        for roi in rtss.StructureSetROISequence:
            roi_number = roi.ROINumber
            roi_name = roi.ROIName.strip().lower()
            roi_names[roi_number] = roi_name
            if roi_name not in label_mapping.values():
                label_mapping[str(len(label_mapping))] = roi_name

        # 创建标签图像[3](@ref)
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

                    # 使用SimpleITK绘制多边形填充（此处需要实现2D/3D填充算法）
                    # 注意：此处需要实现具体的轮廓填充逻辑，以下为简化示例
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

        # 添加到训练列表[1](@ref)
        training_list.append({
            "image": f"./imagesTr/{patient_id}_0000.nii.gz",
            "label": f"./labelsTr/{patient_id}.nii.gz"
        })

    # 生成dataset.json[1,3](@ref)
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


# 使用示例
if __name__ == "__main__":
    convert_dicom_to_nnunet(
        root_dir=r"D:\DevData\data",
        output_dir=r"D:\DevData\raw",
        dataset_id=1,
        dataset_name="Lung_Tumors"
    )