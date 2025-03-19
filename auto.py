import os
import time

from totalsegmentator.python_api import totalsegmentator
# from my_python_api import totalsegmentator

if __name__ == '__main__':
    start = time.time()
    dcm = r"D:\rs\raw_dcm"
    task = "head_glands_cavities"
    os.environ["TOTALSEG_HOME_DIR"] = r"D:\rs"
    save_dir = r"E:\work"
    totalsegmentator(input=dcm, task=task, output=save_dir, output_type="dicom", device="gpu")
    print("cost time: ", time.time() - start)