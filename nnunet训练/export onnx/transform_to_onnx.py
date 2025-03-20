import os
from tqdm import tqdm
from my_nnunet import nnUNetv2_predict
from get_id_config import get_task_details
import global_dir


if __name__ == '__main__':
    # 所有id
    # task_ids = [ 291, 292, 293, 294, 295,299,775,850, 851]
    task_ids = [850, 851]
    # task_ids.extend([297, 300, 732, 733])

    # 配置下载的文件路径
    config_file = os.path.join(global_dir.save_dir,"nnunet","results")
    os.environ["nnUNet_results"] = config_file
    os.environ["home"] = os.path.dirname(config_file)

    # 遍历操作（重写 & 加载 & 转换 & 导出 模型）
    for task_id in tqdm(task_ids, desc="模型id", unit="task_id", position=0):
        task, resample, trainer, crop, model, folds = get_task_details(task_id)
        nnUNetv2_predict(task_id, model=model, folds=[0],
                         trainer=trainer, tta=False,
                         num_threads_preprocessing=3, num_threads_nifti_save=2,
                         plans="nnUNetPlans", device="cuda", quiet=False, step_size=0.5)
