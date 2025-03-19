import os
import pydicom
import nibabel as nib
import numpy as np


def dcm_to_nii_corrected(dcm_folder):
    for root, dirs, files in os.walk(dcm_folder):
        dcm_files = [f for f in files if f.endswith('.dcm')]

        if not dcm_files:
            continue

        dicoms = []
        for f in dcm_files:
            try:
                dicom_file = pydicom.dcmread(os.path.join(root, f))
                if hasattr(dicom_file, 'pixel_array'):
                    dicoms.append(dicom_file)
            except Exception as e:
                print(f"读取文件 {f} 失败: {e}")

        if not dicoms:
            print(f"在文件夹 {root} 中没有有效的 DICOM 文件。")
            continue

        pixel_data = [d.pixel_array for d in dicoms]

        if not pixel_data:
            print(f"在文件夹 {root} 中没有有效的像素数据。")
            continue

        pixel_data = np.stack(pixel_data, axis=-1)

        # 手动调整图像数据方向（交换轴以纠正旋转）
        pixel_data = np.rot90(pixel_data, k=-1, axes=(0, 1))

        # 获取DICOM的方向信息和像素间距
        orientation = dicoms[0].ImageOrientationPatient
        position = dicoms[0].ImagePositionPatient
        pixel_spacing = dicoms[0].PixelSpacing
        slice_thickness = getattr(dicoms[0], 'SliceThickness', 1.0)  # 默认厚度

        row_cosines = np.array(orientation[:3])
        col_cosines = np.array(orientation[3:])
        normal_vector = np.cross(row_cosines, col_cosines)

        # 构造仿射矩阵
        affine = np.eye(4)
        affine[:3, 0] = row_cosines * pixel_spacing[0]
        affine[:3, 1] = col_cosines * pixel_spacing[1]
        affine[:3, 2] = normal_vector * slice_thickness
        affine[:3, 3] = position

        nii_img = nib.Nifti1Image(pixel_data, affine)

        parent_folder = os.path.basename(root)
        grandparent_folder = os.path.basename(os.path.dirname(root))
        nii_filename = f"{grandparent_folder}_nasop_{parent_folder}.nii.gz"
        nii_path = os.path.join(root, nii_filename)

        nib.save(nii_img, nii_path)
        print(f"NIfTI文件已保存到: {nii_path}")

        for dcm_file in dcm_files:
            os.remove(os.path.join(root, dcm_file))
        print("旧的DICOM文件已删除。")


if __name__ == '__main__':
    # 使用示例
    dcm_folder = 'D:\Work\data\WXL01'  # 替换为你的DICOM文件路径
    dcm_to_nii_corrected(dcm_folder)

