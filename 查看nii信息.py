import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

file = r"D:\Work\data\CT_img\0408_nasop_20131204.nii.gz"
data = nib.load(file)
affine = data.affine
header = data.header
print(affine)
# print(header)

img = data.get_fdata()
print(img.shape)
print(img.dtype)
# plt.imshow(img, cmap='gray')
# plt.show()