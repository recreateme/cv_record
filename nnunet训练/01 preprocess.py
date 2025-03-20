import os
import re
from datetime import datetime

def process_subfolders(folder_path):
    """
    处理指定文件夹下的子文件夹，保留日期最新的文件夹（如果子文件夹数量大于1）
    :param folder_path: 指定文件夹的路径
    """
    # 获取指定文件夹下的所有子文件夹
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    if len(subfolders) <= 1:
        return  # 如果子文件夹数量小于等于1，不做任何处理

    date_pattern = re.compile(r'\d{4}[-_]?\d{2}[-_]?\d{2}')
    folder_dates = {}
    for subfolder in subfolders:
        # 从子文件夹名称中提取日期信息
        match = date_pattern.search(subfolder)
        if match:
            date_str = match.group().replace('-', '').replace('_', '')
            try:
                # 将日期字符串转换为datetime对象
                folder_date = datetime.strptime(date_str, '%Y%m%d')
                folder_dates[subfolder] = folder_date
            except ValueError:
                continue

    if not folder_dates:
        return  # 如果没有找到有效的日期信息，不做任何处理

    # 按日期降序排序
    sorted_folders = sorted(folder_dates.items(), key=lambda x: x[1], reverse=True)
    latest_folder = sorted_folders[0][0]

    # 删除除最新文件夹之外的其他文件夹
    for subfolder in subfolders:
        if subfolder != latest_folder:
            folder_to_delete = os.path.join(folder_path, subfolder)
            try:
                # 递归删除文件夹及其内容
                import shutil
                shutil.rmtree(folder_to_delete)
                print(f"Deleted folder: {folder_to_delete}")
            except Exception as e:
                print(f"Error deleting folder {folder_to_delete}: {e}")

def main(root_path):
    """
    主程序，对指定路径下的所有文件夹调用process_subfolders函数进行处理
    :param root_path: 指定的根路径
    """
    # 获取根路径下的所有文件夹
    folders = [f for f in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, f))]
    for folder in folders:
        folder_path = os.path.join(root_path, folder)
        process_subfolders(folder_path)

if __name__ == "__main__":
    # 请将以下路径替换为你实际要处理的根路径
    root_path = r"D:\DevData\SDC003_200"
    main(root_path)