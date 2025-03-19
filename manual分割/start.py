import copy
import json
import os
import sys
import tempfile
import time
from pathlib import Path
import gc

import argparse
import nibabel as nib
import numpy as np
import onnxruntime as ort
import torch
from scipy import ndimage
from torch.nn import functional as F
from batchgenerators.utilities.file_and_folder_operations import load_json, save_pickle
from nibabel.nifti1 import Nifti1Image
from nnunetv2.inference.export_prediction import convert_predicted_logits_to_segmentation_with_correct_shape
from nnunetv2.inference.export_prediction import export_prediction_from_logits
from nnunetv2.inference.sliding_window_prediction import compute_gaussian, compute_steps_for_sliding_window
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

from map2binary import map_taskid_to_partname_ct, map_taskid_to_partname_headneck_muscles, map_taskid_to_partname_mr
from utils import as_closest_canonical, undo_canonical, crop_to_mask, dcm_to_nifti, save_mask_as_rtstruct, \
    combine_masks, check_if_shape_and_affine_identical, change_spacing, \
    add_label_map_to_nifti, keep_largest_blob_multilabel, remove_small_blobs_multilabel, remove_auxiliary_labels
from map2binary import class_map, class_map_5_parts, class_map_parts_mr, class_map_parts_headneck_muscles
from tqdm import tqdm


def load_json(file: str):
    with open(file, 'r') as f:
        a = json.load(f)
    return a


def empty_cache(device: torch.device):
    # 释放空余内存
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    elif device.type == 'mps':
        from torch import mps
        mps.empty_cache()
    else:
        pass


def get_config(task):
    crop_addon = [3, 3, 3]
    if task == "total":
        task_id = [291, 292, 293, 294, 295]
        resample = 1.5
        trainer = "nnUNetTrainerNoMirroring"
        crop = None
        model = "3d_fullres"
        folds = [0]
    elif task == "total_mr":
        task_id = [730, 731]
        resample = 1.5
        trainer = "nnUNetTrainer_DASegOrd0_NoMirroring"
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
    elif task == "coronary_arteries":
        task_id = 503
        resample = None
        trainer = "nnUNetTrainer"
        crop = ["heart"]
        model = "3d_fullres"
        folds = [0]
        print("WARNING: The coronary artery model does not work very robustly. Use with care!")
    elif task == "body":
        task_id = 299
        resample = 1.5
        trainer = "nnUNetTrainer"
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
        crop = ["clavicula_left", "clavicula_right", "vertebrae_C1", "vertebrae_C5", "vertebrae_T1", "vertebrae_T4"]
        crop_addon = [40, 40, 40]
        model = "3d_fullres_high"
        folds = [0]
    return task_id, resample, crop, crop_addon, model, folds


def global_nii(file_in, tmp_dir):
    dcm_to_nifti(file_in, tmp_dir / "converted_dcm.nii.gz", tmp_dir)
    return tmp_dir / "converted_dcm.nii.gz"


def converted_data(resample, crop, crop_addon, tmp_dir):
    img_in_orig = nib.load(os.path.join(tmp_dir, "converted_dcm.nii.gz"))
    if len(img_in_orig.shape) == 2:
        raise ValueError("TotalSegmentator does not work for 2D images. Use a 3D image.")
    if len(img_in_orig.shape) > 3:
        img_in_orig = nib.Nifti1Image(img_in_orig.get_fdata()[:, :, :, 0], img_in_orig.affine)

    img_in = nib.Nifti1Image(img_in_orig.get_fdata(), img_in_orig.affine)  # 创建原件的副本
    # 如果需要裁剪，则进行裁剪
    bbox = None
    if crop is not None:
        crop_mask_img = crop
        img_in, bbox = crop_to_mask(img_in_orig, crop_mask_img, addon=crop_addon, dtype=np.int32)

    img_in = as_closest_canonical(img_in)
    # 重采样
    if resample is not None:
        img_in_rsp = change_spacing(img_in, resample, order=3, dtype=np.int32, nr_cpus=1)
    else:
        img_in_rsp = img_in

    converted_img_path = tmp_dir / "converted.nii.gz"
    nib.save(img_in_rsp, converted_img_path)
    return img_in, img_in_rsp, converted_img_path, bbox


def compute_steps_for_sliding_window(image_size, tile_size, tile_step_size):
    target_step_sizes_in_voxels = np.array(tile_size) * tile_step_size
    num_steps = np.ceil((np.array(image_size) - np.array(tile_size)) / target_step_sizes_in_voxels).astype(int) + 1
    max_step_values = np.array(image_size) - np.array(tile_size)
    actual_step_sizes = np.where(num_steps > 1, max_step_values / (num_steps - 1), 0)
    steps = [np.round(actual_step_sizes[i] * np.arange(num_steps[i])).astype(int) for i in range(len(tile_size))]
    return steps


def get_slicers(image_size, patch_size, tile_step_size):
    steps = compute_steps_for_sliding_window(image_size, patch_size, tile_step_size)
    slicers = []
    for sx in steps[0]:
        for sy in steps[1]:
            for sz in steps[2]:
                slicers.append(
                    tuple(
                        [slice(sx, sx + patch_size[0]), slice(sy, sy + patch_size[1]), slice(sz, sz + patch_size[2])]))
    return slicers


def build_trainer(config_dir, model='3d_fullres', tid=None):
    dataset_json_file = os.path.join(config_dir, 'dataset.json')
    plan_file = os.path.join(config_dir, 'plans.json')
    model_path = os.path.join(config_dir, f"task_{tid}.onnx")

    dataset_json = load_json(dataset_json_file)
    plan = load_json(plan_file)
    plans_manager = PlansManager(plan)
    configuration_manager = plans_manager.get_configuration(model)

    try:
        session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider'])
        print("使用gpu进行推理")
    except (RuntimeError, ValueError):
        session = ort.InferenceSession(model_path)
        print("使用cpu进行推理")

    return session, plans_manager, configuration_manager, dataset_json


def export_prediction_from_logits(predicted_array_or_file, properties_dict, configuration_manager, plans_manager,
                                  dataset_json_dict_or_file, save_probabilities: bool = False):
    if isinstance(dataset_json_dict_or_file, str):
        dataset_json_dict_or_file = load_json(dataset_json_dict_or_file)

    label_manager = plans_manager.get_label_manager(dataset_json_dict_or_file)
    ret = convert_predicted_logits_to_segmentation_with_correct_shape(
        predicted_array_or_file, plans_manager, configuration_manager, label_manager, properties_dict,
        return_probabilities=save_probabilities
    )
    del predicted_array_or_file
    segmentation_final = ret
    del ret
    segmentation_final = segmentation_final.transpose((2, 1, 0)).astype(np.uint8)
    return segmentation_final


def predict_fn(session, data, slicers, config_manager, output_channels, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.as_tensor(data, dtype=torch.float32, device=device)
    n_predictions = torch.zeros(data.shape[1:], dtype=torch.float16, device=device)
    predicted_logits = torch.zeros((output_channels, *data.shape[1:]), dtype=torch.float16, device=device)

    gaussian = compute_gaussian(
        tuple(config_manager.patch_size),
        sigma_scale=1. / 8,
        value_scaling_factor=10,
        device=device
    )

    with torch.no_grad():
        for sl in tqdm(slicers, desc="Predicting"):
            try:
                workon = data[:, sl[0], sl[1], sl[2]][None, :]
                input_name = session.get_inputs()[0].name
                output = session.run(None, {input_name: workon.cpu().numpy()})
                prediction = torch.as_tensor(output[0][0], dtype=torch.float16, device=device)
                prediction *= gaussian

                new_slicers = (slice(None),) + tuple(sl)
                predicted_logits[new_slicers] += prediction
                n_predictions[sl] += gaussian

            except Exception as e:
                print(f"Error processing slice {sl}: {e}")
                continue

            finally:
                torch.cuda.empty_cache()
                gc.collect()

    n_predictions = torch.clamp(n_predictions, min=1e-8)
    predicted_logits /= n_predictions

    return predicted_logits


def get_seg_combined(args):
    file_in = args.get('file_in')
    file_out = args.get('file_out')
    img_type = args.get('img_type')
    resample = args.get('resample')
    crop_addon = args.get('crop_addon')
    tmp_dir = args.get('tmp_dir')
    crop = args.get('crop')
    img_in, img_in_rsp, converted_img_path, bbox = converted_data(resample=resample, crop=crop, crop_addon=crop_addon,
                                                                  tmp_dir=tmp_dir)

    task_name = args.get('task_name')
    task_ids = args.get('task_id', [])
    task_ids = [task_ids] if not isinstance(task_ids, list) else task_ids
    device = args.get('device') or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = args.get('base_dir')

    class_map_parts = args.get('class_map_parts')
    map_taskid_to_partname = args.get('map_taskid_to_partname')

    if task_name == "total" and resample is not None and resample[0] < 3.0:
        step_size = 0.8
    else:
        step_size = 0.5
    model = args.get('model')

    class_map_inv = {v: k for k, v in class_map[task_name].items()}
    seg_combined = np.zeros(img_in_rsp.shape, dtype=np.uint8)
    muti_model = len(task_ids) > 1
    for idx, tid in enumerate(task_ids):
        try:
            model_folder = os.path.join(base_dir, str(tid))
            session, plans_manager, configuration_manager, dataset_json = build_trainer(model_folder, model, tid)
            preprocessor = configuration_manager.preprocessor_class()
            data, seg, data_properties = preprocessor.run_case(
                [converted_img_path], None, plans_manager,
                configuration_manager, dataset_json
            )

            output_channels = plans_manager.get_label_manager(dataset_json).num_segmentation_heads
            # 确认padding
            data = torch.as_tensor(data, dtype=torch.float32, device=device)
            data, slicer_revert_padding = pad_nd_image(data, configuration_manager.patch_size, 'constant', {'value': 0},
                                                       True, None)
            slicers = get_slicers(data.shape[1:], configuration_manager.patch_size, step_size)

            prediction = predict_fn(session, data, slicers, configuration_manager, output_channels, device).to("cpu")

            # undo padding
            prediction = prediction[(slice(None), *slicer_revert_padding[1:])].numpy()

            print(f"完成第{idx + 1}个任务的预测")
            seg = export_prediction_from_logits(
                prediction, data_properties, configuration_manager,
                plans_manager, dataset_json)

            if muti_model:
                for jdx, class_name in class_map_parts[map_taskid_to_partname[tid]].items():
                    seg_combined[seg == jdx] = class_map_inv[class_name]
            else:
                seg_combined = seg

        except Exception as e:
            print(f"Error processing task {tid}: {e}")

        finally:
            torch.cuda.empty_cache()
            gc.collect()
    return nib.Nifti1Image(seg_combined, data_properties['nibabel_stuff']['reoriented_affine']), img_in, bbox


def get_args(task_name, file_in, file_out, base_dir, tmp_dir):
    task_id, resample, crop, crop_addon, model, _ = get_config(task_name)

    if isinstance(resample, float):
        resample = [resample, resample, resample]
    class_map_parts = None
    map_taskid_to_partname = None
    if task_name == "total":
        class_map_parts = class_map_5_parts
        map_taskid_to_partname = map_taskid_to_partname_ct
    elif task_name == "total_mr":
        class_map_parts = class_map_parts_mr
        map_taskid_to_partname = map_taskid_to_partname_mr
    elif task_name == "headneck_muscles":
        class_map_parts = class_map_parts_headneck_muscles
        map_taskid_to_partname = map_taskid_to_partname_headneck_muscles

    if torch.cuda.is_available():
        torch.set_num_threads(1)
        device = torch.device('cuda')
    else:
        import multiprocessing
        torch.set_num_threads(multiprocessing.cpu_count())
        device = torch.device('cpu')

    args_dict = {
        "task_name": task_name,
        "file_in": file_in,
        "file_out": file_out,
        "base_dir": base_dir,
        'task_id': task_id,
        'resample': resample,
        'crop': crop,
        'crop_addon': crop_addon,
        "model": model,
        'class_map_parts': class_map_parts,
        'map_taskid_to_partname': map_taskid_to_partname,
        "device": device,
        "tmp_dir": tmp_dir
    }
    return args_dict


def keep_largest_blob_multilabel(data, class_map, rois, debug=False, quiet=False):
    class_map_inv = {v: k for k, v in class_map.items()}
    for roi in tqdm(rois, disable=quiet):
        idx = class_map_inv[roi]
        data_roi = data == idx
        cleaned_roi = keep_largest_blob(data_roi, debug) > 0.5
        data[data_roi] = 0  # Clear the original ROI in data
        data[cleaned_roi] = idx  # Write back the cleaned ROI into data
    return data


def get_resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)


def undo_crop(img, ref_img, bbox):
    img_out = np.zeros(ref_img.shape)
    img_out[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1], bbox[2][0]:bbox[2][1]] = img.get_fdata()
    return nib.Nifti1Image(img_out, ref_img.affine)


def keep_largest_blob(data, debug=False):
    blob_map, nr_of_blobs = ndimage.label(data)
    counts = [np.sum(blob_map == i) for i in range(1, nr_of_blobs + 1)]  # this will not count background
    if len(counts) == 0: return data  # no foreground
    largest_blob_label = np.argmax(counts) + 1  # +1 because labels start from 1
    if debug: print(f"size of largest blob: {np.max(counts)}")
    return (blob_map == largest_blob_label).astype(np.uint8)


def remove_small_blobs(img: np.ndarray, interval=[10, 30], debug=False) -> np.ndarray:
    mask, number_of_blobs = ndimage.label(img)
    if debug: print('Number of blobs before: ' + str(number_of_blobs))
    counts = np.bincount(mask.flatten())  # number of pixels in each blob

    if len(counts) <= 1: return img

    remove = np.where((counts <= interval[0]) | (counts > interval[1]), True, False)
    remove_idx = np.nonzero(remove)[0]
    mask[np.isin(mask, remove_idx)] = 0
    mask[mask > 0] = 1

    if debug:
        print(f"counts: {sorted(counts)[::-1]}")
        _, number_of_blobs_after = ndimage.label(mask)
        print('Number of blobs after: ' + str(number_of_blobs_after))

    return mask


def remove_small_blobs_multilabel(data, class_map, rois, interval=[10, 30], debug=False, quiet=False):
    class_map_inv = {v: k for k, v in class_map.items()}
    for roi in tqdm(rois, disable=quiet):
        idx = class_map_inv[roi]
        data_roi = (data == idx)
        cleaned_roi = remove_small_blobs(data_roi, interval, debug) > 0.5  # Remove small blobs from this ROI
        data[data_roi] = 0
        data[cleaned_roi] = idx
    return data


def post_process(seg_combined_nii, args_dict, img_in, converted_files, bbox=None):
    task_name = args_dict.get("task_name")
    img_pred = remove_auxiliary_labels(seg_combined_nii, task_name)
    if task_name == "body":
        img_pred_pp = keep_largest_blob_multilabel(img_pred.get_fdata().astype(np.uint8),
                                                   class_map[task_name], ["body_trunc"], debug=False)
        img_pred = nib.Nifti1Image(img_pred_pp, img_pred.affine)

    if task_name == "body":
        vox_vol = np.prod(img_pred.header.get_zooms())
        size_thr_mm3 = 50000 / vox_vol
        img_pred_pp = remove_small_blobs_multilabel(img_pred.get_fdata().astype(np.uint8),
                                                    class_map[task_name], ["body_extremities"],
                                                    interval=[size_thr_mm3, 1e10], debug=False)
        img_pred = nib.Nifti1Image(img_pred_pp, img_pred.affine)

    resample = args_dict.get("resample")
    if resample is not None:
        img_pred = change_spacing(img_pred, resample, img_in.shape, order=0, dtype=np.uint8, nr_cpus=1,
                                  force_affine=img_in.affine)
    print("Undoing canonical...")
    img_in_orig = nib.load(converted_files)
    img_pred = undo_canonical(img_pred, img_in_orig)

    # undo crop
    crop = args_dict.get("crop")
    if crop is not None:
        img_pred = undo_crop(img_pred, img_in_orig, bbox)
    check_if_shape_and_affine_identical(img_in_orig, img_pred)

    img_data = img_pred.get_fdata().astype(np.uint8)

    label_map = class_map[task_name]
    new_header = img_in_orig.header.copy()
    new_header.set_data_dtype(np.uint8)
    img_out = nib.Nifti1Image(img_data, img_pred.affine, new_header)
    img_out = add_label_map_to_nifti(img_out, label_map)

    return img_out


def save_img(img_data, file_in, file_out, task_name):
    print("Saving segmentations...")
    selected_classes = class_map[task_name]
    file_out = Path(file_out)
    file_in = Path(file_in)
    file_out.mkdir(exist_ok=True, parents=True)
    empty_cache(torch.device("cpu"))
    import gc
    gc.collect()
    img_data = img_data.get_fdata()
    save_mask_as_rtstruct(img_data, selected_classes, file_in, file_out / "segmentations.dcm")


def get_mask(crop_task, organ_seg, crop):
    class_map_inv = {v: k for k, v in class_map[crop_task].items()}
    crop_mask = np.zeros(organ_seg.shape, dtype=np.uint8)
    organ_seg_data = organ_seg.get_fdata()
    # roi_subset_crop = [map_to_total[roi] if roi in map_to_total else roi for roi in roi_subset]
    for roi in crop:
        crop_mask[organ_seg_data == class_map_inv[roi]] = 1
        nums = np.array(crop_mask).astype(np.uint8).sum()
    crop_mask = nib.Nifti1Image(crop_mask, organ_seg.affine)

    return crop_mask


def pad_nd_image(image, new_shape, mode="constant", kwargs=None, return_slicer=False, shape_must_be_divisible_by=None):
    if kwargs is None:
        kwargs = {}

    old_shape = np.array(image.shape)

    if shape_must_be_divisible_by is not None:
        assert isinstance(shape_must_be_divisible_by, (int, list, tuple, np.ndarray))
        if isinstance(shape_must_be_divisible_by, int):
            shape_must_be_divisible_by = [shape_must_be_divisible_by] * len(image.shape)
        else:
            if len(shape_must_be_divisible_by) < len(image.shape):
                shape_must_be_divisible_by = [1] * (len(image.shape) - len(shape_must_be_divisible_by)) + \
                                             list(shape_must_be_divisible_by)

    if new_shape is None:
        assert shape_must_be_divisible_by is not None
        new_shape = image.shape
    if len(new_shape) < len(image.shape):
        new_shape = list(image.shape[:len(image.shape) - len(new_shape)]) + list(new_shape)
    new_shape = [max(new_shape[i], old_shape[i]) for i in range(len(new_shape))]
    if shape_must_be_divisible_by is not None:
        if not isinstance(shape_must_be_divisible_by, (list, tuple, np.ndarray)):
            shape_must_be_divisible_by = [shape_must_be_divisible_by] * len(new_shape)

        if len(shape_must_be_divisible_by) < len(new_shape):
            shape_must_be_divisible_by = [1] * (len(new_shape) - len(shape_must_be_divisible_by)) + \
                                         list(shape_must_be_divisible_by)

        for i in range(len(new_shape)):
            if new_shape[i] % shape_must_be_divisible_by[i] == 0:
                new_shape[i] -= shape_must_be_divisible_by[i]

        new_shape = np.array([new_shape[i] + shape_must_be_divisible_by[i] - new_shape[i] %
                              shape_must_be_divisible_by[i] for i in range(len(new_shape))])

    difference = new_shape - old_shape
    pad_below = difference // 2
    pad_above = difference // 2 + difference % 2
    pad_list = [list(i) for i in zip(pad_below, pad_above)]

    if not ((all([i == 0 for i in pad_below])) and (all([i == 0 for i in pad_above]))):
        if isinstance(image, np.ndarray):
            res = np.pad(image, pad_list, mode, **kwargs)
        elif isinstance(image, torch.Tensor):
            # torch padding has the weirdest interface ever. Like wtf? Y u no read numpy documentation? So much easier
            torch_pad_list = [i for j in pad_list for i in j[::-1]][::-1]
            res = F.pad(image, torch_pad_list, mode, **kwargs)
    else:
        res = image

    if not return_slicer:
        return res
    else:
        pad_list = np.array(pad_list)
        pad_list[:, 1] = np.array(res.shape) - pad_list[:, 1]
        slicer = tuple(slice(*i) for i in pad_list)
        return res, slicer


def main():
    start = time.time()
    print("Start......")
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, default="total")
    parser.add_argument("--file_in", type=str, help="待处理dicom文件夹")
    parser.add_argument("--file_out", type=str, help="输出Rtstructure路径")
    args = parser.parse_args()
    #
    task_name = args.task_name
    file_in = args.file_in
    file_out = args.file_out
    # task_name = "head_glands_cavities"
    # file_in = r"D:\DEV\AutoContour\data"
    # file_out = r"D:\DEV\AutoContour"
    base_dir = get_resource_path("results")
    # base_dir = r"D:\DEV\AutoContour\results"
    if not os.path.exists(base_dir):
        raise ValueError("依赖模型文件不存在")
    print(f"本次任务为：{task_name}, 输入文件为：{file_in}")

    with tempfile.TemporaryDirectory(prefix="nnunet_tmp_") as tmp_folder:
        tmp_dir = Path(tmp_folder)
        converted_files = global_nii(file_in, tmp_dir)
        main_args = get_args(task_name, file_in, file_out, base_dir, tmp_dir)
        crop = main_args.get("crop")
        if crop is not None:
            crop_task = "total_mr" if task_name == "total_mr" else "total"
            body_config = get_args(crop_task, file_in, file_out, base_dir, tmp_dir)
            crop_model_task = 733 if crop_task == "total_mr" else 298
            crop_spacing = 6.0
            body_config['resample'] = [crop_spacing, crop_spacing, crop_spacing]
            body_config['task_id'] = crop_model_task

            seg_combined_nii, img_in, _ = get_seg_combined(body_config)
            # rs = np.array(seg_combined_nii.get_fdata()).astype(np.int8)
            # cnt = np.sum(rs)
            # rs = np.where(rs > 0, 1, 0)
            # cnt2 = np.sum(rs)

            organ_seg = post_process(seg_combined_nii, body_config, img_in, converted_files)  # undo shape
            mask = get_mask(crop_task, organ_seg, crop)

            # rs = np.array(mask.get_fdata()).astype(np.int8)
            # cnt1 = np.sum(rs)
            # rs = np.where(rs > 0, 1, 0)
            # cnt2 = np.sum(rs)

            main_args.update({"crop": mask})
            main_args.update({"crop_addon": [20, 20, 20]})

            del organ_seg, mask, seg_combined_nii, img_in

        # 当前任务分割
        seg_combined_nii, img_in, bbox = get_seg_combined(main_args)
        # rs = np.array(seg_combined_nii.get_fdata()).astype(np.int8)
        # cnt1 = np.sum(rs)
        # rs = np.where(rs > 0, 1, 0)
        # cnt2 = np.sum(rs)

        img_data = post_process(seg_combined_nii, main_args, img_in, converted_files, bbox)
        save_img(img_data, file_in, file_out, task_name)
        # 正式处理
        # print(time.time() - start)
        # print("Over")


if __name__ == '__main__':
    main()
