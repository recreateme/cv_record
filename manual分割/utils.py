import multiprocessing
import os
import zipfile
from datetime import time

import dicom2nifti
import nibabel as nib
import numpy as np
from joblib import Parallel, delayed
from scipy import ndimage
from tqdm import tqdm

from map2binary import class_map_5_parts,class_map


def as_closest_canonical(img_in):
    return nib.as_closest_canonical(img_in)


def undo_canonical(img_can, img_orig):
    from nibabel.orientations import io_orientation, axcodes2ornt, ornt_transform
    img_ornt = io_orientation(img_orig.affine)
    ras_ornt = axcodes2ornt("RAS")
    to_canonical = img_ornt
    from_canonical = ornt_transform(ras_ornt, img_ornt)
    return img_can.as_reoriented(from_canonical)

def get_bbox_from_mask(mask, outside_value=-900, addon=0):
    if type(addon) is int:
        addon = [addon] * 3
    if (mask > outside_value).sum() == 0:
        print("WARNING: Could not crop because no foreground detected")
        minzidx, maxzidx = 0, mask.shape[0]
        minxidx, maxxidx = 0, mask.shape[1]
        minyidx, maxyidx = 0, mask.shape[2]
    else:
        mask_voxel_coords = np.where(mask > outside_value)
        minzidx = int(np.min(mask_voxel_coords[0])) - addon[0]
        maxzidx = int(np.max(mask_voxel_coords[0])) + 1 + addon[0]
        minxidx = int(np.min(mask_voxel_coords[1])) - addon[1]
        maxxidx = int(np.max(mask_voxel_coords[1])) + 1 + addon[1]
        minyidx = int(np.min(mask_voxel_coords[2])) - addon[2]
        maxyidx = int(np.max(mask_voxel_coords[2])) + 1 + addon[2]

    # Avoid bbox to get out of image size
    s = mask.shape
    minzidx = max(0, minzidx)
    maxzidx = min(s[0], maxzidx)
    minxidx = max(0, minxidx)
    maxxidx = min(s[1], maxxidx)
    minyidx = max(0, minyidx)
    maxyidx = min(s[2], maxyidx)

    return [[minzidx, maxzidx], [minxidx, maxxidx], [minyidx, maxyidx]]

def crop_to_bbox(image, bbox):
    assert len(image.shape) == 3, "only supports 3d images"
    return image[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1], bbox[2][0]:bbox[2][1]]

def crop_to_bbox_nifti(image: nib.Nifti1Image, bbox, dtype=None) -> nib.Nifti1Image:
    assert len(image.shape) == 3, "only supports 3d images"
    data = image.get_fdata()

    # Crop the image
    data_cropped = crop_to_bbox(data, bbox)

    # Update the affine matrix
    affine = np.copy(image.affine)
    affine[:3, 3] = np.dot(affine, np.array([bbox[0][0], bbox[1][0], bbox[2][0], 1]))[:3]

    data_type = image.dataobj.dtype if dtype is None else dtype
    return nib.Nifti1Image(data_cropped.astype(data_type), affine)

def crop_to_mask(img_in, mask_img, addon=[0,0,0], dtype=None, verbose=False):
    mask = mask_img.get_fdata()

    addon = (np.array(addon) / img_in.header.get_zooms()).astype(int)  # mm to voxels
    bbox = get_bbox_from_mask(mask, outside_value=0, addon=addon)

    img_out = crop_to_bbox_nifti(img_in, bbox, dtype)
    return img_out, bbox


def dcm_to_nifti(input_path, output_path, tmp_dir=None, verbose=False):
    """
    Uses dicom2nifti package (also works on windows)

    input_path: a directory of dicom slices or a zip file of dicom slices
    output_path: a nifti file path
    tmp_dir: extract zip file to this directory, else to the same directory as the zip file
    """
    # Check if input_path is a zip file and extract it
    if zipfile.is_zipfile(input_path):
        if verbose: print(f"Extracting zip file: {input_path}")
        extract_dir = os.path.splitext(input_path)[0] if tmp_dir is None else tmp_dir / "extracted_dcm"
        with zipfile.ZipFile(input_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
            input_path = extract_dir

    # Convert to nifti
    dicom2nifti.dicom_series_to_nifti(input_path, output_path, reorient_nifti=True)


def save_mask_as_rtstruct(img_data, selected_classes, dcm_reference_file, output_path):
    """
    dcm_reference_file: a directory with dcm slices ??
    """
    from rt_utils import RTStructBuilder
    import logging
    logging.basicConfig(level=logging.WARNING)  # avoid messages from rt_utils

    # create new RT Struct - requires original DICOM
    rtstruct = RTStructBuilder.create_new(dicom_series_path=dcm_reference_file)

    # add mask to RT Struct
    for class_idx, class_name in tqdm(selected_classes.items()):
        binary_img = img_data == class_idx
        if binary_img.sum() > 0:  # only save none-empty images

            # rotate nii to match DICOM orientation
            binary_img = np.rot90(binary_img, 1, (0, 1))  # rotate segmentation in-plane

            # add segmentation to RT Struct
            rtstruct.add_roi(
                mask=binary_img,  # has to be a binary numpy array
                name=class_name
            )

    rtstruct.save(str(output_path))

def combine_masks(mask_dir, class_type):
    """
    Combine classes to masks

    mask_dir: directory of totalsegmetator masks
    class_type: ribs | vertebrae | vertebrae_ribs | lung | heart

    returns: nibabel image
    """
    if class_type == "ribs":
        masks = list(class_map_5_parts["class_map_part_ribs"].values())
    elif class_type == "vertebrae":
        masks = list(class_map_5_parts["class_map_part_vertebrae"].values())
    elif class_type == "vertebrae_ribs":
        masks = list(class_map_5_parts["class_map_part_vertebrae"].values()) + list(class_map_5_parts["class_map_part_ribs"].values())
    elif class_type == "lung":
        masks = ["lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right",
                 "lung_middle_lobe_right", "lung_lower_lobe_right"]
    elif class_type == "lung_left":
        masks = ["lung_upper_lobe_left", "lung_lower_lobe_left"]
    elif class_type == "lung_right":
        masks = ["lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right"]
    elif class_type == "pelvis":
        masks = ["femur_left", "femur_right", "hip_left", "hip_right"]
    elif class_type == "body":
        masks = ["body_trunc", "body_extremities"]

    ref_img = None
    for mask in masks:
        if (mask_dir / f"{mask}.nii.gz").exists():
            ref_img = nib.load(mask_dir / f"{masks[0]}.nii.gz")
        else:
            raise ValueError(f"Could not find {mask_dir / mask}.nii.gz. Did you run TotalSegmentator successfully?")

    combined = np.zeros(ref_img.shape, dtype=np.uint8)
    for idx, mask in enumerate(masks):
        if (mask_dir / f"{mask}.nii.gz").exists():
            img = nib.load(mask_dir / f"{mask}.nii.gz").get_fdata()
            combined[img > 0.5] = 1

    return nib.Nifti1Image(combined, ref_img.affine)

def check_if_shape_and_affine_identical(img_1, img_2):

    max_diff = np.abs(img_1.affine - img_2.affine).max()
    if max_diff > 1e-5:
        print("Affine in:")
        print(img_1.affine)
        print("Affine out:")
        print(img_2.affine)
        print("Diff:")
        print(np.abs(img_1.affine-img_2.affine))
        print("WARNING: Output affine not equal to input affine. This should not happen.")

    if img_1.shape != img_2.shape:
        print("Shape in:")
        print(img_1.shape)
        print("Shape out:")
        print(img_2.shape)
        print("WARNING: Output shape not equal to input shape. This should not happen.")

def add_label_map_to_nifti(img_in, label_map):
    """
    This will save the information which label in a segmentation mask has which name to the extended header.

    img: nifti image
    label_map: a dictionary with label ids and names | a list of names and a running id will be generated starting at 1

    returns: nifti image
    """
    data = img_in.get_fdata()

    if label_map is None:
        label_map = {idx+1: f"L{val}" for idx, val in enumerate(np.unique(data)[1:])}

    if type(label_map) is not dict:   # can be list or dict_values list
        label_map = {idx+1: val for idx, val in enumerate(label_map)}

    colors = [[255,0,0],[0,255,0],[0,0,255],[255,255,0],[255,0,255],[0,255,255],[255,128,0],[255,0,128],[128,255,128],[0,128,255],[128,128,128],[185,170,155]]
    xmlpre = '<?xml version="1.0" encoding="UTF-8"?> <CaretExtension>  <Date><![CDATA[2013-07-14T05:45:09]]></Date>   <VolumeInformation Index="0">   <LabelTable>'

    body = ''
    for label_id, label_name in label_map.items():
        rgb = colors[label_id%len(colors)]
        body += f'<Label Key="{label_id}" Red="{rgb[0]/255}" Green="{rgb[1]/255}" Blue="{rgb[2]/255}" Alpha="1"><![CDATA[{label_name}]]></Label>\n'

    xmlpost = '  </LabelTable>  <StudyMetaDataLinkSet>  </StudyMetaDataLinkSet>  <VolumeType><![CDATA[Label]]></VolumeType>   </VolumeInformation></CaretExtension>'
    xml = xmlpre + "\n" + body + "\n" + xmlpost + "\n              "

    img_in.header.extensions.append(nib.nifti1.Nifti1Extension(0,bytes(xml,'utf-8')))

    return img_in

def keep_largest_blob(data, debug=False):
    blob_map, nr_of_blobs = ndimage.label(data)
    # Get number of pixels in each blob
    # counts = list(np.bincount(blob_map.flatten()))  # this will also count background -> bug
    counts = [np.sum(blob_map == i) for i in range(1, nr_of_blobs + 1)]  # this will not count background
    if len(counts) == 0: return data  # no foreground
    largest_blob_label = np.argmax(counts) + 1  # +1 because labels start from 1
    if debug: print(f"size of largest blob: {np.max(counts)}")
    return (blob_map == largest_blob_label).astype(np.uint8)

def keep_largest_blob_multilabel(data, class_map, rois, debug=False, quiet=False):
    """
    Keep the largest blob for the classes defined in rois.

    data: multilabel image (np.array)
    class_map: class map {label_idx: label_name}
    rois: list of labels where to filter for the largest blob

    return multilabel image (np.array)
    """
    st = time.time()
    class_map_inv = {v: k for k, v in class_map.items()}
    for roi in tqdm(rois, disable=quiet):
        idx = class_map_inv[roi]
        data_roi = data == idx
        cleaned_roi = keep_largest_blob(data_roi, debug) > 0.5
        data[data_roi] = 0   # Clear the original ROI in data
        data[cleaned_roi] = idx   # Write back the cleaned ROI into data
    # print(f"  keep_largest_blob_multilabel took {time.time() - st:.2f}s")
    return data

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
    st = time.time()
    class_map_inv = {v: k for k, v in class_map.items()}

    for roi in tqdm(rois, disable=quiet):
        idx = class_map_inv[roi]
        data_roi = (data == idx)
        cleaned_roi = remove_small_blobs(data_roi, interval, debug) > 0.5  # Remove small blobs from this ROI
        data[data_roi] = 0
        data[cleaned_roi] = idx
    return data

def remove_auxiliary_labels(img, task_name):
    task_name_aux = task_name + "_auxiliary"
    if task_name_aux in class_map:
        class_map_aux = class_map[task_name_aux]
        data = img.get_fdata()
        # remove auxiliary labels
        for idx in class_map_aux.keys():
            data[data == idx] = 0
        return nib.Nifti1Image(data.astype(np.uint8), img.affine)
    else:
        return img

def resample_img(img, zoom=0.5, order=0, nr_cpus=-1):
    """
    img: [x,y,z,(t)]
    zoom: 0.5 will halfen the image resolution (make image smaller)

    Resize numpy image array to new size.

    Faster than resample_img_nnunet.
    Resample_img_nnunet maybe slightly better quality on CT (but not sure).

    Works for 2D and 3D and 4D images.
    """
    def _process_gradient(grad_idx):
        return ndimage.zoom(img[:, :, :, grad_idx], zoom, order=order)

    dim = len(img.shape)

    # Add dimensions to make each input 4D
    if dim == 2:
        img = img[..., None, None]
    if dim == 3:
        img = img[..., None]

    nr_cpus = multiprocessing.cpu_count() if nr_cpus == -1 else nr_cpus
    img_sm = Parallel(n_jobs=nr_cpus)(delayed(_process_gradient)(grad_idx) for grad_idx in range(img.shape[3]))
    img_sm = np.array(img_sm).transpose(1, 2, 3, 0)  # grads channel was in front -> put to back
    # Remove added dimensions
    # img_sm = img_sm[:,:,:,0] if img_sm.shape[3] == 1 else img_sm  # remove channel dim if only 1 element
    if dim == 3:
        img_sm = img_sm[:,:,:,0]
    if dim == 2:
        img_sm = img_sm[:,:,0,0]
    return img_sm


def change_spacing(img_in, new_spacing=1.25, target_shape=None, order=0, nr_cpus=1,
                   nnunet_resample=False, dtype=None, remove_negative=False, force_affine=None):
    data = img_in.get_fdata()  # quite slow
    old_shape = np.array(data.shape)
    img_spacing = np.array(img_in.header.get_zooms())

    if len(img_spacing) == 4:
        img_spacing = img_spacing[:3]  # for 4D images only use spacing of first 3 dims

    if type(new_spacing) is float:
        new_spacing = [new_spacing,] * 3   # for 3D and 4D
    new_spacing = np.array(new_spacing)

    if len(old_shape) == 2:
        img_spacing = np.array(list(img_spacing) + [new_spacing[2],])

    if target_shape is not None:
        zoom = np.array(target_shape) / old_shape
        new_spacing = img_spacing / zoom
    else:
        zoom = img_spacing / new_spacing

    if np.array_equal(img_spacing, new_spacing):
        return img_in

    new_affine = np.copy(img_in.affine)
    new_affine = np.copy(img_in.affine)
    new_affine[:3, 0] = new_affine[:3, 0] / zoom[0]
    new_affine[:3, 1] = new_affine[:3, 1] / zoom[1]
    new_affine[:3, 2] = new_affine[:3, 2] / zoom[2]

    new_data = resample_img(data, zoom=zoom, order=order, nr_cpus=nr_cpus)  # cpu resampling
    if remove_negative:
        new_data[new_data < 1e-4] = 0

    if dtype is not None:
        new_data = new_data.astype(dtype)

    if force_affine is not None:
        new_affine = force_affine

    return nib.Nifti1Image(new_data, new_affine)
