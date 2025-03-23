import os
import pydicom
import SimpleITK as sitk
import numpy as np
from typing import List, Dict
from skimage.draw import polygon


def convert_dicom_rtstruct_to_nnunet(
        dicom_dir,
        rtstruct_path,
        image_path,
        label_path,
        label_dtype: type = np.uint16
) -> None:
    dicom_series = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(dicom_dir)
    if not dicom_series:
        raise ValueError(f"No valid DICOM series found in {dicom_dir}")
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(dicom_series)
    image = reader.Execute()
    rtss = pydicom.dcmread(rtstruct_path)
    ref_image_uid = rtss.ReferencedFrameOfReferenceSequence[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence[
        0].SeriesInstanceUID
    if not _validate_coordinate_system(image, rtss):
        raise RuntimeError("DICOM image and RTSTRUCT coordinate system mismatch!")
    mask_array = _rtstruct_to_mask(
        rtss=rtss,
        reference_image=image,
        label_dtype=label_dtype
    )

    # 保存图像
    sitk.WriteImage(image, image_path)
    # 保存标签 (nnUNet格式: [patient_id].nii.gz)
    mask_image = sitk.GetImageFromArray(mask_array)
    mask_image.CopyInformation(image)  # 关键：复制空间信息
    sitk.WriteImage(mask_image, label_path)


def _validate_coordinate_system(image: sitk.Image, rtss: pydicom.Dataset) -> bool:
    """验证DICOM图像与RTSTRUCT坐标系一致性"""
    # 获取参考图像UID (RTSTRUCT中记录的)
    ref_series_uid = rtss.ReferencedFrameOfReferenceSequence[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence[
        0].SeriesInstanceUID

    # 获取实际图像的UID
    reader = sitk.ImageFileReader()
    reader.SetFileName(image.GetMetaData("0020|000e"))  # Series Instance UID
    actual_series_uid = reader.GetMetaData("0020|000e")

    return ref_series_uid == actual_series_uid


def _rtstruct_to_mask(
        rtss: pydicom.Dataset,
        reference_image: sitk.Image,
        label_dtype: type
) -> np.ndarray:
    """核心函数：将RTSTRUCT转换为标签掩码"""
    # 初始化空标签矩阵
    image_array = sitk.GetArrayFromImage(reference_image)  # (Z, Y, X)
    mask = np.zeros_like(image_array, dtype=label_dtype)

    # 获取ROI名称映射表
    roi_info = {
        roi.ROINumber: {
            "name": roi.ROIName,
            "color": rtss.ROIContourSequence[roi.ROINumber - 1].ROIDisplayColor
        } for roi in rtss.StructureSetROISequence
    }

    # 遍历所有ROI
    for roi_contour in rtss.ROIContourSequence:
        roi_number = roi_contour.ReferencedROINumber
        if roi_number not in roi_info:
            continue

        print(f"Processing ROI {roi_number}: {roi_info[roi_number]['name']}...")

        # 遍历该ROI的所有轮廓
        for contour in roi_contour.ContourSequence:
            # 转换坐标点 (DICOM LPS -> SimpleITK XYZ)
            points = np.array(contour.ContourData).reshape(-1, 3)
            physical_points = [tuple(map(float, p)) for p in points]

            # 转换为体素坐标
            indices = [
                reference_image.TransformPhysicalPointToContinuousIndex(p)
                for p in physical_points
            ]

            # 验证是否为平面轮廓 (必须位于同一Z层)
            z_coords = [round(idx[2]) for idx in indices]
            if len(set(z_coords)) != 1:
                print(f"警告: ROI {roi_number} 的轮廓跨越多个切片，已跳过")
                continue

            z_slice = int(round(z_coords[0]))

            # 生成多边形坐标 (注意Y/X顺序)
            y_coords = [idx[1] for idx in indices]
            x_coords = [idx[0] for idx in indices]

            try:
                rr, cc = polygon(y_coords, x_coords, shape=mask.shape[1:])
                valid_mask = (rr >= 0) & (rr < mask.shape[1]) & (cc >= 0) & (cc < mask.shape[2])
                rr = rr[valid_mask]
                cc = cc[valid_mask]

                # 写入标签 (允许多标签重叠，后处理的覆盖先处理的值)
                mask[z_slice, rr, cc] = roi_number
            except Exception as e:
                print(f"ROI {roi_number} 轮廓处理失败: {str(e)}")

    return mask


# 使用示例
convert_dicom_rtstruct_to_nnunet(
    dicom_dir="data/dcm",
    rtstruct_path="data/meta/RS1.3.6.1.4.1.2452.6.2667341226.1244394865.4006930326.1855796959.dcm",
    output_dir="data/nnUNet_formatted",
    patient_id="case_001",
    label_dtype=np.uint16
)