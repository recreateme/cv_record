import argparse
import os
import shutil
import sys

import SimpleITK as sitk
import numpy as np
import torch
import torchvision.transforms as transforms
from monai.transforms import NormalizeIntensity, Resize

from Convert_nii import dcm_to_nii_corrected
from model import *


def list_of_strings(arg):
    return arg.split(',')


def resource_path(weights, relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, weights, relative_path)
    return os.path.join(os.path.abspath("."), weights, relative_path)


def get_parent_directory(file_path, levels=2):
    parent_dir = file_path
    for _ in range(levels):
        parent_dir = os.path.dirname(parent_dir)
    return parent_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_CT', type=str, help='base directory for CT', default="CT_img")
    parser.add_argument('--base_MR', type=str, help='base directory for MR', default="MR_img")
    # parser.add_argument('--baseline_CT', type=str, help='filename of the baseline CT slices')
    # parser.add_argument('--baseline_MR', type=str, help='filename of the baseline MR slices')
    parser.add_argument('--followup_CT', type=str, help='filename of the followup CT slices')
    parser.add_argument('--followup_MR', type=str, help='filename of the followup MR slices')

    parser.add_argument('--model', type=str, help='Rad-D or Rad-S', default="Rad-S")
    parser.add_argument('--Differentiation', type=int,
                        help='Histological type, 0 for high differentiation, 1 for moderately differentiation,'
                             ' 2 for low differentiation, and 3 for undifferentiation', default=1)
    parser.add_argument('--NASH', type=int, help='non-alcoholic steatohepatitis or non-alcoholic steatohepatitis,'
                                                 ' 0 for no, and 1 for yes', default=0)
    parser.add_argument('--Surgery', type=int, help='0 for non-surgery and 1 for surgery', default=0)
    parser.add_argument('--PVT', type=int, help='partial or complete portal vein tumor thrombosis,'
                                                ' 0 for no, and 1 for yes', default=0)
    parser.add_argument('--EBRT', type=int, help='external beam radiation therapy,'
                                                 ' 0 for no, and 1 for yes', default=0)
    parser.add_argument('--TACE', type=int, help='transarterial embolization or transarterial chemoembolization,'
                                                 ' 0 for no, and 1 for yes', default=0)
    parser.add_argument('--RFAMWA', type=int, help='radiofrequency ablation or microwave ablation,'
                                                   ' 0 for no, and 1 for yes', default=0)
    args = parser.parse_args()

    transform = transforms.Compose([
        Resize(spatial_size=(224, 224)),
        NormalizeIntensity()
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_CT = dcm_to_nii_corrected(args.base_CT)
    base_MR = dcm_to_nii_corrected(args.base_MR)

    # base_CT = r"D:\Work\data\CT_img\0408_nasop_20131204.nii.gz"
    # base_MR =r"D:\Work\data\MR_img\0408_nasop_20131126.nii.gz"

    x_liver, x_lung, x_liver_1, x_lung_1 = [], [], [], []

    img = sitk.GetArrayFromImage(sitk.ReadImage(base_CT))
    for img_temp in range(img.ndim):
        if transform:
            img = transform(img)
            x_liver.append(img[img_temp])
    x_liver = torch.stack(x_liver).unsqueeze(0)

    img = sitk.GetArrayFromImage(sitk.ReadImage(base_MR))
    for img_temp in range(img.ndim):
        if transform:
            img = transform(img)
            x_lung.append(img[img_temp])
    x_lung = torch.stack(x_lung).unsqueeze(0)

    if args.model == "Rad-D":
        img = sitk.GetArrayFromImage(sitk.ReadImage(args.followup_CT))
        for img_temp in range(img.ndim):
            if transform:
                img = transform(img)
                x_liver_1.append(img[img_temp])
        x_liver_1 = torch.stack(x_liver_1).unsqueeze(0)

        img = sitk.GetArrayFromImage(sitk.ReadImage(args.followup_MR))
        for img_temp in range(img.ndim):
            if transform:
                img = transform(img)
                x_lung_1.append(img[img_temp])

        x_lung_1 = torch.stack(x_lung_1).unsqueeze(0)
        model = PrognosisModelD().to(device)
        model_path = resource_path("weights", "rad-D-pretrained.pth")
        model.load_state_dict(torch.load(model_path), map_location=torch.device('cpu'))
        model.eval()
        out = model(x_liver.to(device), x_lung.to(device), x_liver_1.to(device), x_lung_1.to(device))
    elif args.model == "Rad-S":
        model = PrognosisModelS().to(device)
        model_path = resource_path("weights", "rad-S-pretrained.pth")
        model.load_state_dict(torch.load(model_path))
        model.eval()
        out = model(x_liver.to(device), x_lung.to(device))
    else:
        raise NotImplementedError

    radiology_score = abs(out.item())
    clinical_score = 0.3747 * args.Differentiation + 0.1593 * args.NASH - 0.1801 * args.Surgery + \
                     0.6732 * args.PVT - 0.8235 * args.EBRT + 0.6482 * args.TACE - 0.4497 * args.RFAMWA
    final_risk = np.log(9.883337 * radiology_score + 0.530013 * clinical_score)
    tp = get_parent_directory(base_CT, 2)
    with open(os.path.join(tp, "risk_score.txt"), "w", encoding="utf-8") as f:
        f.write(f"baseline 0.66\n")
        f.write(f"risk_score: {final_risk}")
        # shutil.rmtree(get_parent_directory(base_CT,1))
        # shutil.rmtree(get_parent_directory(base_MR,1))
