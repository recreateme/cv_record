def get_task_details(task_id):
    task_mapping = {
        291: ("total", 1.5, "nnUNetTrainerNoMirroring", None, "3d_fullres", [0]),
        292: ("total", 1.5, "nnUNetTrainerNoMirroring", None, "3d_fullres", [0]),
        293: ("total", 1.5, "nnUNetTrainerNoMirroring", None, "3d_fullres", [0]),
        294: ("total", 1.5, "nnUNetTrainerNoMirroring", None, "3d_fullres", [0]),
        295: ("total", 1.5, "nnUNetTrainerNoMirroring", None, "3d_fullres", [0]),
        297: ("total", 3.0, "nnUNetTrainer_4000epochs_NoMirroring", None, "3d_fullres", [0]),
        298: ("total", 6.0, "nnUNetTrainer_4000epochs_NoMirroring", None, "3d_fullres", [0]),
        730: ("total_mr", 1.5, "nnUNetTrainer_DASegOrd0_NoMirroring", None, "3d_fullres", [0]),
        731: ("total_mr", 1.5, "nnUNetTrainer_DASegOrd0_NoMirroring", None, "3d_fullres", [0]),
        732: ("total_mr", 3.0, "nnUNetTrainer_DASegOrd0_NoMirroring", None, "3d_fullres", [0]),
        733: ("total_mr", 6.0, "nnUNetTrainer_DASegOrd0_NoMirroring", None, "3d_fullres", [0]),
        258: ("lung_vessels", None, "nnUNetTrainer",
              ["lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right",
               "lung_middle_lobe_right", "lung_lower_lobe_right"], "3d_fullres", [0]),
        150: ("cerebral_bleed", None, "nnUNetTrainer", ["brain"], "3d_fullres", [0]),
        260: ("hip_implant", None, "nnUNetTrainer",
              ["femur_left", "femur_right", "hip_left", "hip_right"], "3d_fullres", [0]),
        503: ("coronary_arteries", None, "nnUNetTrainer", ["heart"], "3d_fullres", [0]),
        299: ("body", 1.5, "nnUNetTrainer", None, "3d_fullres", [0]),
        300: ("body", 6.0, "nnUNetTrainer", None, "3d_fullres", [0]),
        315: ("pleural_pericard_effusion", None, "nnUNetTrainer",
              ["lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right",
               "lung_middle_lobe_right", "lung_lower_lobe_right"], "3d_fullres", None),
        8: ("liver_vessels", None, "nnUNetTrainer", ["liver"], "3d_fullres", [0]),
        775: ("head_glands_cavities", [0.75, 0.75, 1.0], "nnUNetTrainer_DASegOrd0_NoMirroring",
              ["skull"], "3d_fullres_high", [0]),
        776: ("headneck_bones_vessels", [0.75, 0.75, 1.0], "nnUNetTrainer_DASegOrd0_NoMirroring",
              ["clavicula_left", "clavicula_right", "vertebrae_C1", "vertebrae_C5", "vertebrae_T1", "vertebrae_T4"],
              "3d_fullres_high", [0]),
        777: ("head_muscles", [0.75, 0.75, 1.0], "nnUNetTrainer_DASegOrd0_NoMirroring",
              ["skull"], "3d_fullres_high", [0]),
        778: ("headneck_muscles", [0.75, 0.75, 1.0], "nnUNetTrainer_DASegOrd0_NoMirroring",
              ["clavicula_left", "clavicula_right", "vertebrae_C1", "vertebrae_C5", "vertebrae_T1", "vertebrae_T4"],
              "3d_fullres_high", [0]),
        779: ("headneck_muscles", [0.75, 0.75, 1.0], "nnUNetTrainer_DASegOrd0_NoMirroring",
              ["clavicula_left", "clavicula_right", "vertebrae_C1", "vertebrae_C5", "vertebrae_T1", "vertebrae_T4"],
              "3d_fullres_high", [0]),
        850:("total_mr",1.5, "nnUNetTrainer_2000epochs_NoMirroring", None, "3d_fullres", [0]),
        851: ("total_mr",1.5, "nnUNetTrainer_2000epochs_NoMirroring", None, "3d_fullres", [0])
    }
    return task_mapping.get(task_id)


if __name__ == "__main__":
    # Example usage
    task_id = 778
    details = get_task_details(task_id)
    if details:
        task, resample, trainer, crop, model, folds = details
        print(f"Task: {task}, Resample: {resample}, Trainer: {trainer}, Crop: {crop}, Model: {model}, Folds: {folds}")
    else:
        print("Invalid task_id")
