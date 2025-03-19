import nibabel as nib
import numpy as np
import onnxruntime as ort
import torch

# 加载 NIfTI 文件
# nii_file = r'D:\rs\测试\s01_0000_origin.nii.gz'
# img = nib.load(nii_file)
# img_data = img.get_fdata()
# 假设模型需要输入形状为 (1, C, H, W)
# img_data = np.expand_dims(img_data, axis=0)
# img_data = img_data.astype(np.float32)
#
# mean = np.array([0.5])
# std = np.array([0.5])
# img_data = (img_data - mean) / std

# onnx_model_path = r'D:\rs\测试\my_nnunet02.onnx'
# session = ort.InferenceSession(onnx_model_path)
#
# input_name = session.get_inputs()[0].name
# output_name = session.get_outputs()[0].name
# shape = (1, 1, 128, 128, 128)
# img_in = img_data[:128,:128,:128]
# # 为img_in在前面个添加两个维度
# img_in = np.expand_dims(img_in, axis=0)
# img_in = np.expand_dims(img_in, axis=0).astype(np.float32)
#
#
# output = session.run([output_name], {input_name: img_in})
#
# output_data = output[0]

# 创建NIfTI 图像并保存
# output_nii = nib.Nifti1Image(output_data[0], img.affine)
# nib.save(output_nii, r'D:\output.nii.gz')
#
# print("推理完成，输出已保存。")
print(ort.get_device())