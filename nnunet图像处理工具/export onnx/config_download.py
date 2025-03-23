import os

import tqdm
import global_dir

if __name__ == '__main__':

    os.environ["TOTALSEG_HOME_DIR"] = global_dir.save_dir       # 模型保存目录
    # tasks = ["total", "total_mr", "lung_vessels", "cerebral_bleed", "hip_implant", "coronary_arteries",
    #          "body", "pleural_pericard_effusion", "liver_vessels", "head_glands_cavities","headneck_bones_vessels",
    #           "head_muscles,", "headneck_muscles"]
    tasks = ["total","total_mr", "body","head_glands_cavities"]
    
    for task in tqdm.tqdm(tasks):
        task_id = None
        if task == "total":
            task_id = [291, 292, 293, 294, 295]
            resample = 1.5
            trainer = "nnUNetTrainerNoMirroring"
            crop = None
            model = "3d_fullres"
            folds = [0]
        elif task == "total_mr":
            task_id = [850, 851]
            resample = 1.5
            trainer = "nnUNetTrainer_2000epochs_NoMirroring"
            crop = None
            model = "3d_fullres"
            folds = [0]
        elif task == "lung_vessels":
            task_id = 258
            resample = None
            trainer = "nnUNetTrainer"
            crop = ["lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right",
                    "lung_middle_lobe_right", "lung_lower_lobe_right"]
            model = "3d_fullres"
            folds = [0]
        elif task == "cerebral_bleed":
            task_id = 150
            resample = None
            trainer = "nnUNetTrainer"
            crop = ["brain"]
            model = "3d_fullres"
            folds = [0]
        elif task == "hip_implant":
            task_id = 260
            resample = None
            trainer = "nnUNetTrainer"
            crop = ["femur_left", "femur_right", "hip_left", "hip_right"]
            model = "3d_fullres"
            folds = [0]
        elif task == "body":
            task_id = 299
            resample = 1.5
            trainer = "nnUNetTrainer"
            crop = None
            model = "3d_fullres"
            folds = [0]
        elif task == "body_mr":
            task_id = 597
            resample = 1.5
            trainer = "nnUNetTrainer_DASegOrd0"
            crop = None
            model = "3d_fullres"
            folds = [0]
        elif task == "vertebrae_mr":
            task_id = 756
            resample = None
            trainer = "nnUNetTrainer_DASegOrd0_NoMirroring"
            crop = None
            model = "3d_fullres"
            folds = [0]
        elif task == "pleural_pericard_effusion":
            task_id = 315
            resample = None
            trainer = "nnUNetTrainer"
            crop = ["lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right",
                    "lung_middle_lobe_right", "lung_lower_lobe_right"]
            crop_addon = [50, 50, 50]
            model = "3d_fullres"
            folds = None
        elif task == "liver_vessels":
            task_id = 8
            resample = None
            trainer = "nnUNetTrainer"
            crop = ["liver"]
            crop_addon = [20, 20, 20]
            model = "3d_fullres"
            folds = [0]
        elif task == "head_glands_cavities":
            task_id = 775
            resample = [0.75, 0.75, 1.0]
            trainer = "nnUNetTrainer_DASegOrd0_NoMirroring"
            crop = ["skull"]
            crop_addon = [10, 10, 10]
            model = "3d_fullres_high"
            folds = [0]
        elif task == "headneck_bones_vessels":
            task_id = 776
            resample = [0.75, 0.75, 1.0]
            trainer = "nnUNetTrainer_DASegOrd0_NoMirroring"
            # crop = ["skull", "clavicula_left", "clavicula_right", "vertebrae_C5", "vertebrae_T1", "vertebrae_T4"]
            # crop_addon = [10, 10, 10]
            crop = ["clavicula_left", "clavicula_right", "vertebrae_C1", "vertebrae_C5", "vertebrae_T1", "vertebrae_T4"]
            crop_addon = [40, 40, 40]
            model = "3d_fullres_high"
            folds = [0]
        elif task == "head_muscles":
            task_id = 777
            resample = [0.75, 0.75, 1.0]
            trainer = "nnUNetTrainer_DASegOrd0_NoMirroring"
            crop = ["skull"]
            crop_addon = [10, 10, 10]
            model = "3d_fullres_high"
            folds = [0]
        elif task == "headneck_muscles":
            task_id = [778, 779]
            resample = [0.75, 0.75, 1.0]
            trainer = "nnUNetTrainer_DASegOrd0_NoMirroring"
            # crop = ["skull", "clavicula_left", "clavicula_right", "vertebrae_C5", "vertebrae_T1", "vertebrae_T4"]
            # crop_addon = [10, 10, 10]
            crop = ["clavicula_left", "clavicula_right", "vertebrae_C1", "vertebrae_C5", "vertebrae_T1", "vertebrae_T4"]
            crop_addon = [40, 40, 40]
            model = "3d_fullres_high"
            folds = [0]
        elif task == "oculomotor_muscles":
            task_id = 351
            resample = [0.47251562774181366, 0.47251562774181366, 0.8500002026557922]
            trainer = "nnUNetTrainer_DASegOrd0_NoMirroring"
            crop = ["skull"]
            crop_addon = [20, 20, 20]
            model = "3d_fullres"
            folds = [0]
        elif task == "lung_nodules":
            task_id = 913
            resample = [1.5, 1.5, 1.5]
            trainer = "nnUNetTrainer_MOSAIC_1k_QuarterLR_NoMirroring"
            crop = ["lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right",
                    "lung_middle_lobe_right", "lung_lower_lobe_right"]
            crop_addon = [10, 10, 10]
            model = "3d_fullres"
            folds = [0]
        elif task == "kidney_cysts":
            task_id = 789
            resample = [1.5, 1.5, 1.5]
            trainer = "nnUNetTrainer_DASegOrd0_NoMirroring"
            crop = ["kidney_left", "kidney_right", "liver", "spleen", "colon"]
            crop_addon = [10, 10, 10]
            model = "3d_fullres"
            folds = [0]
        elif task == "breasts":
            task_id = 527
            resample = [1.5, 1.5, 1.5]
            trainer = "nnUNetTrainer_DASegOrd0_NoMirroring"
            crop = None
            model = "3d_fullres"
            folds = [0]
        elif task == "ventricle_parts":
            task_id = 552
            resample = [1.0, 0.4345703125, 0.4384765625]
            trainer = "nnUNetTrainerNoMirroring"
            crop = ["brain"]
            crop_addon = [0, 0, 0]
            model = "3d_fullres"
            folds = [0]
        elif task == "liver_segments":
            task_id = 570
            resample = [1.5, 0.8046879768371582, 0.8046879768371582]
            trainer = "nnUNetTrainerNoMirroring"
            crop = ["liver"]
            crop_addon = [10, 10, 10]
            model = "3d_fullres"
            folds = [0]
        elif task == "liver_segments_mr":
            task_id = 576
            resample = [3.0, 1.1875, 1.1250001788139343]
            trainer = "nnUNetTrainer_DASegOrd0_NoMirroring"
            crop = ["liver"]
            crop_addon = [10, 10, 10]
            model = "3d_fullres"
            folds = [0]

        if task_id is None:
            print(f"Task {task} not found")
            continue
        if type(task_id) is list:
            for tid in task_id:
                download_pretrained_weights(tid)
        else:
            download_pretrained_weights(task_id)
