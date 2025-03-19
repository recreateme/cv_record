import numpy as np
import sys


# 创建一个非连续数组
arr = np.arange(12).reshape(3, 4)[::2]
print(arr.flags['C_CONTIGUOUS'])  # 输出 False，表示数组不是连续的
print(sys.getsizeof(arr))


# 使用 ascontiguousarray 转换为连续数组
cont_arr = np.ascontiguousarray(arr)
print(cont_arr.flags['C_CONTIGUOUS'])  # 输出 True，表示数组是连续的
print(sys.getsizeof(cont_arr))

print(arr.__array_interface__)  # 输出非连续数组的内存地址
print(cont_arr.__array_interface__)  # 输出连续数组的内存地址
