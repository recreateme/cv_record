import os
import numpy as np
import pydicom
import nibabel as nib

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

        pixel_data = np.rot90(pixel_data, k=-1, axes=(0, 1))

        orientation = dicoms[0].ImageOrientationPatient  # 图像方向
        position = dicoms[0].ImagePositionPatient  # 图像位置
        pixel_spacing = dicoms[0].PixelSpacing  # 像素间距
        slice_thickness = getattr(dicoms[0], 'SliceThickness', 1.0)  # 切片厚度，默认值为 1.0

        # 计算行方向和列方向的余弦向量
        row_cosines = np.array(orientation[:3])
        col_cosines = np.array(orientation[3:])
        # 计算法向量
        normal_vector = np.cross(row_cosines, col_cosines)

        # 构造仿射矩阵
        affine = np.eye(4)  # 初始化 4x4 单位矩阵
        affine[:3, 0] = row_cosines * pixel_spacing[0]  # 设置行方向的缩放
        affine[:3, 1] = col_cosines * pixel_spacing[1]  # 设置列方向的缩放
        affine[:3, 2] = normal_vector * slice_thickness  # 设置切片方向的缩放
        affine[:3, 3] = position  # 设置图像位置

        # 创建 NIfTI 图像对象
        nii_img = nib.Nifti1Image(pixel_data, affine)

        # 生成 NIfTI 文件名
        parent_folder = os.path.basename(root)  # 当前文件夹名称
        grandparent_folder = os.path.basename(os.path.dirname(root))  # 父文件夹名称
        nii_filename = f"{grandparent_folder}_nasop_{parent_folder}.nii.gz"  # 文件名格式
        nii_path = os.path.join(root, nii_filename)  # 文件保存路径

        nib.save(nii_img, nii_path)
        # print(f"NIfTI文件已保存到: {nii_path}")

        # for dcm_file in dcm_files:
        #     os.remove(os.path.join(root, dcm_file))
        return nii_path