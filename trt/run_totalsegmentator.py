# import subprocess
# import sys
# import os
#
#
# def main():
#     # 检查输入参数
#     if len(sys.argv) != 3:
#         print("用法: python run_totalsegmentator.py <输入文件> <输出目录>")
#         sys.exit(1)
#
#     input_file = sys.argv[1]
#     output_dir = sys.argv[2]
#
#     # 检查文件和目录是否存在
#     if not os.path.isfile(input_file):
#         print(f"输入文件不存在: {input_file}")
#         sys.exit(1)
#
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#
#     # 构造 Docker 命令
#     command = [
#         "docker", "run", "--gpus", "device=0", "-v", f"{os.path.dirname(input_file)}:/tmp",
#         "wasserth/totalsegmentator:2.2.1", "TotalSegmentator",
#         "-i", f"/tmp/{os.path.basename(input_file)}",
#         "-o", f"/tmp/segmentations"
#     ]

#     # 执行命令
#     try:
#         subprocess.run(command, check=True)
#         print("处理完成，结果保存在:", output_dir)
#     except subprocess.CalledProcessError as e:
#         print(f"命令执行失败: {e}")
#         sys.exit(1)
#
#
# if __name__ == "__main__":
#     main()


import torch
print(torch.cuda.is_available())