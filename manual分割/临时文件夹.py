# import os
# import tempfile
#
# # 创建一个临时文件夹
# with tempfile.TemporaryDirectory() as temp_dir:
#     print(f"创建的临时文件夹路径: {temp_dir}")
#
#     # 在临时文件夹中创建一个文件
#     file_path = os.path.join(temp_dir, 'temp_file.txt')
#     with open(file_path, 'w') as file:
#         file.write("这是一些临时文件的内容")
#
#     # 读取文件内容
#     with open(file_path, 'r') as file:
#         content = file.read()
#         print(f"文件内容: {content}")
#
# # 代码块执行完毕后，临时文件夹及其内容会被自动删除
# print("临时文件夹已被删除")

# 如果村子啊crop，需要线执行以便粗糙的分割

x=["clavicula_left", "clavicula_right", "vertebrae_C1", "vertebrae_C5", "vertebrae_T1", "vertebrae_T4"]
print(type(x[0]))