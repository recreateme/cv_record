import os
import subprocess
import tempfile
import zipfile
import nibabel as nib
import dicom2nifti


def dcm_to_nifti(input_path, output_path, tmp_dir=None, verbose=False):
    if zipfile.is_zipfile(input_path):
        if verbose: print(f"Extracting zip file: {input_path}")
        extract_dir = os.path.splitext(input_path)[0] if tmp_dir is None else tmp_dir / "extracted_dcm"
        with zipfile.ZipFile(input_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
            input_path = extract_dir
    dicom2nifti.dicom_series_to_nifti(input_path, output_path, reorient_nifti=True)
    return output_path


if __name__ == '__main__':
    file_in = r"D:\rs\Head\WXL01"
    converted_nii = "ct.nii.gz"
    output_dir = os.path.dirname(file_in)

    with tempfile.TemporaryDirectory(prefix="nnunet_tmp_") as tmp_folder:
        nii_path = dcm_to_nifti(file_in, os.path.join(tmp_folder, converted_nii), verbose=True)

        input_nii_file = nii_path  # 输入的.nii.gz文件
        image = nib.load(input_nii_file)
        output_drr_file = 'output.drr'  # 输出的DRR图像文件
        proj_geom_file = 'uu.geom'  # 投影几何配置文件

        # 创建投影几何配置文件
        # 注意：以下参数需要根据实际情况进行调整
        with open(proj_geom_file, 'w') as f:
            f.write('projection:\n')
            f.write('  num-projections: 1\n')
            f.write('  num-views: 1\n')
            f.write('  sid: 1500\n')  # 源到探测器的距离
            f.write('  sdd: 1500\n')  # 源到物体的距离
            f.write('  ai: 0\n')  # 源到探测器的角度（弧度）
            f.write('  angle-begin: 0\n')
            f.write('  angle-end: 0\n')
            f.write('  angle-step: 0\n')
            f.write('  det-rows: 512\n')
            f.write('  det-cols: 512\n')
            f.write('  det-pitch: 1.0\n')

        # 使用plastimatch命令行工具生成DRR图像
        # command = [
        #     'plastimatch', 'convert', '--input', input_nii_file,
        #     '--output-drr', output_drr_file,
        #     '--proj-geom', proj_geom_file
        # ]
        # subprocess.run(command, check=True)
        #
        # # 清理生成的配置文件
        # os.remove(proj_geom_file)
        #
        # print(f'DRR image generated: {output_drr_file}')

