import itk

def generate_drr_from_dicom(dicom_dir, output_drr_path):
    # 读取 DICOM 文件
    reader = itk.ImageSeriesReader.New()

    dicom_names = itk.GDCMSeriesFileNames.New()
    dicom_names.SetDirectory(dicom_dir)
    reader.SetFileNames(dicom_names.GetFileNames())
    image = reader.Execute()

    # 定义投影几何
    drr_filter = itk.RayCastProjectionImageFilter.New(image)
    drr_filter.SetProjectionType(itk.RayCastProjectionImageFilter.PERSPECTIVE)  # 透视投影
    drr_filter.SetSourceToDetectorDistance(1000)  # 源到探测器的距离
    drr_filter.SetSourceToObjectDistance(500)     # 源到物体的距离

    # 生成 DRR 图像
    drr_image = drr_filter.Execute()

    # 保存 DRR 图像
    itk.imwrite(drr_image, output_drr_path)
    print(f"DRR 图像已保存到: {output_drr_path}")


if __name__ == '__main__':
    # 输入 DICOM 文件夹路径
    dicom_dir = r"D:\rs\Head\WXL01"

    # 输出 DRR 图像路径
    output_drr_path = r"D:\rs\Head\drr_output.nii.gz"

    # 生成 DRR 图像
    generate_drr_from_dicom(dicom_dir, output_drr_path)