import pandas as pd
import os
import glob
import torch
import cv2
import numpy as np
import torchvision.transforms.functional as TF
import multiprocessing
import math

from torch.utils.data import Dataset
from skimage import io
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from sklearn.preprocessing import OneHotEncoder
from MultiMEDal_multimodal_medical.src.datasets.fast_dataloader import (
    InMemoryFastDataloader,
)


class CBIS_DDSM_dataset(Dataset):
    def __init__(
        self,
        data_root,
        split,
        class_type,
        birads=[0, 1, 2, 3, 4, 5],
        abnormality=["mass", "calcification"],
        transform=None,
    ):
        self.data_root = data_root

        image_root = os.path.join(self.data_root, "CBIS_DDSM_pngs")
        mass_image_root = os.path.join(image_root, "mass", split)
        calc_image_root = os.path.join(image_root, "calc", split)

        desc_file = os.path.join(self.data_root, f"cbis_ddsm_{split}_set.xlsx")

        desc_pd = pd.read_excel(pd.ExcelFile(desc_file))

        desc_pd = desc_pd.loc[desc_pd["assessment"].isin(birads)]
        desc_pd = desc_pd.loc[desc_pd["abnormality type"].isin(abnormality)]

        self.lesion_feats_label = desc_pd.columns[21:].to_numpy()
        print("# Samples:", desc_pd.shape)
        

        if class_type == "2classes":
            self.classes = ["BENIGN", "MALIGNANT"]
        elif class_type == "4classes":
            self.classes = [
                "BENIGN_CALCIFICATION",
                "BENIGN_MASS",
                "MALIGNANT_CALCIFICATION",
                "MALIGNANT_MASS",
            ]

        self.transform = transform

        self.image_paths = []
        self.images = []
        self.lesion_feats = []
        self.labels = []
        self.patients = []

        for idx, row in desc_pd.iterrows():
            # if idx == 256:
            #     break

            cropped_file, pathology = row["cropped image file path"], row["pathology"]
            abnormality_type = row["abnormality type"]
            patient_id = row["patient_id"]

            # abnormality type, mass shape, mass margin
            # calcification type, calcification distribution
            lesion_feat = row[21:].to_numpy(dtype=np.int)

            if pathology == "BENIGN_WITHOUT_CALLBACK":
                pathology = "BENIGN"

            # Image paths
            if abnormality_type == "mass":
                cropped_file_path = os.path.join(mass_image_root, cropped_file)
            elif abnormality_type == "calcification":
                cropped_file_path = os.path.join(calc_image_root, cropped_file)

            # Check Mask ROI v.s. cropped ROI
            p = Path(cropped_file_path)
            cropped_dir = os.path.join(*p.parts[:-3])

            cropped_file_path = None
            cropped_area = float("inf")
            for img_path in glob.glob(os.path.join(cropped_dir, "**", "**", "*.png")):
                img = Image.open(img_path)
                area = img.size[0] * img.size[1]

                if (cropped_file_path is None) or (area < cropped_area):
                    cropped_file_path = img_path
                    cropped_area = area

            # Images
            cropped_filename, _ = os.path.splitext(os.path.basename(cropped_file_path))
            cropped_png_path = os.path.join(
                os.path.dirname(cropped_file_path), f"{cropped_filename}.png"
            )

            self.image_paths.append(cropped_png_path)

            image = Image.open(cropped_png_path)

            norm_image = cv2.normalize(
                np.array(image), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
            )

            # Otsu's thresholding - BEGIN
            # blur = cv2.GaussianBlur(norm_image, ksize=(5, 5), sigmaX=2, sigmaY=2)
            # retval, mask = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            # norm_image = cv2.bitwise_and(norm_image, norm_image, mask=mask)
            # Otsu's thresholding - END

            image = Image.fromarray(norm_image).convert("RGB")

            self.images.append(image)

            # Lesion Feats
            self.lesion_feats.append(lesion_feat)

            # Labels
            if class_type == "2classes":
                label = int(pathology == "MALIGNANT")
            elif class_type == "4classes":
                label = self.classes.index((pathology + "_" + abnormality_type).upper())

            self.labels.append(label)

            self.patients.append(patient_id)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        # image = Image.open(self.image_paths[index])
        # image = image.convert('RGB')
        image = self.images[index]
        image_path = self.image_paths[index]
        lesion_feat = self.lesion_feats[index]
        label = self.labels[index]

        if self.transform:
            image = self.transform(image)

        return dict(
            {
                "image": image,
                "label": label,
                "lesion_feat": lesion_feat,
                "image_path": image_path,
            }
        )

    def get_all_images(self):
        return self.images

    def get_all_lesion_feats(self):
        return self.lesion_feats

    def get_all_labels(self):
        return self.labels

    def get_lesion_feats_label(self):
        return self.lesion_feats_label

    def get_all_patients(self):
        return self.patients


class CBIS_DDSM_whole_mamm_dataset(Dataset):
    def __init__(
        self,
        data_root,
        split,
        class_type,
        birads=[0, 1, 2, 3, 4, 5],
        abnormality=["mass", "calcification"],
        data_dir=None, # "CBIS_DDSM_pngs"
        transform=None,
        get_mask=False
    ):
        self.data_root = data_root
        self.get_mask = get_mask

        image_root = os.path.join(self.data_root, data_dir)
        mass_image_root = os.path.join(image_root, "mass", split)
        calc_image_root = os.path.join(image_root, "calc", split)

        desc_file = os.path.join(self.data_root, f"cbis_ddsm_{split}_set.xlsx")

        desc_pd = pd.read_excel(pd.ExcelFile(desc_file))

        desc_pd = desc_pd.loc[desc_pd["assessment"].isin(birads)]
        desc_pd = desc_pd.loc[desc_pd["abnormality type"].isin(abnormality)]

        self.lesion_feats_label = desc_pd.columns[21:].to_numpy()
        print("# Samples:", desc_pd.shape)
        # print('Columns:', self.lesion_feats_label)

        if class_type == "2classes":
            self.classes = ["BENIGN", "MALIGNANT"]

        self.transform = transform

        self.image_paths = []
        self.images = []
        self.labels = []
        self.patients = []
        self.mask_paths = []

        for idx, row in tqdm(desc_pd.iterrows()):
            # if idx > 500:
            #     break

            mamm_file, pathology = row["image file path"], row["pathology"]
            abnormality_type = row["abnormality type"]
            patient_id = row["patient_id"]

            # abnormality type, mass shape, mass margin
            # calcification type, calcification distribution
            # lesion_feat = row[21:].to_numpy(dtype=np.int)

            if pathology == "BENIGN_WITHOUT_CALLBACK":
                pathology = "BENIGN"

            # Image paths
            if abnormality_type == "mass":
                mamm_file_path = os.path.join(mass_image_root, mamm_file)
            elif abnormality_type == "calcification":
                mamm_file_path = os.path.join(calc_image_root, mamm_file)

            p = Path(mamm_file_path)

            mamm_dir = os.path.join(*p.parts[:-1])

            mamm_file_path = os.path.join(mamm_dir, "000000.png")

            # Images
            self.image_paths.append(mamm_file_path)

            # Labels
            if class_type == "2classes":
                label = int(pathology == "MALIGNANT")

            self.labels.append(label)

            self.patients.append(patient_id)

            # Masks
            all_match_records = desc_pd[desc_pd['image file path'] == mamm_file]
            all_masks = []
            for _idx, _row in all_match_records.iterrows():
                mask_file = _row['ROI mask file path']
            

                if abnormality_type == "mass":
                    mask_file_path = os.path.join(mass_image_root, mask_file)
                elif abnormality_type == "calcification":
                    mask_file_path = os.path.join(calc_image_root, mask_file)

                p = Path(mask_file_path)
                mask_dir = os.path.join(*p.parts[:-3])

                mask_file_path = None
                mask_area = float("inf")

                for img_path in glob.glob(os.path.join(mask_dir, "**", "**", "*.png")):
                    img = Image.open(img_path)
                    area = img.size[0] * img.size[1]

                    if (mask_file_path is None) or (area > mask_area):
                        mask_file_path = img_path
                        mask_area = area

                all_masks.append(mask_file_path)
            
            self.mask_paths.append(all_masks)

        max_num_masks = max(len(mask_paths) for mask_paths in self.mask_paths)

        self.mask_paths = [mask_paths + [''] * (max_num_masks - len(mask_paths)) for mask_paths in self.mask_paths]

        # # Multiprocess Loading
        # proc = InMemoryFastDataloader(img_size=512)
        # pool = multiprocessing.Pool()
        # self.images = pool.map(proc, self.image_paths)

        # Check label at mammogram-level (For 2 classes case)
        # If a mammogram contain at least one cancerous lesion, then label is 'malignant'.
        # Otherwise, label will be 'benign'
        cancer_check = defaultdict(set)

        for idx, img_path in enumerate(self.image_paths):
            cancer_check[img_path].add(self.labels[idx])

        for idx, img_path in enumerate(self.image_paths):
            self.labels[idx] = sum(cancer_check[self.image_paths[idx]])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        image_path = self.image_paths[index]

        image = Image.open(image_path)
        
        mask_path = self.mask_paths[index]        
        
        label = self.labels[index]

        if self.transform:
            image = self.transform(image)            

        if self.get_mask:
            return dict({"image": image, "label": label, "image_path": image_path, "mask_path": mask_path})
        else:
            return dict({"image": image, "label": label, "image_path": image_path})

    def get_all_images(self):
        return self.image_paths
    
    def get_all_masks(self):
        return self.mask_paths

    def get_all_lesion_feats(self):
        return self.lesion_feats

    def get_all_labels(self):
        return self.labels

    def get_lesion_feats_label(self):
        return self.lesion_feats_label

    def get_all_patients(self):
        return self.patients


class CBIS_DDSM_whole_mamm_breast_density_dataset(Dataset):
    def __init__(
        self,
        data_root,
        split,
        birads=[0, 1, 2, 3, 4, 5],
        abnormality=["mass", "calcification"],
        data_dir=None, # "CBIS_DDSM_pngs"
        transform=None,
        get_mask=False
    ):
        self.data_root = data_root
        self.get_mask = get_mask

        image_root = os.path.join(self.data_root, data_dir)
        mass_image_root = os.path.join(image_root, "mass", split)
        calc_image_root = os.path.join(image_root, "calc", split)

        desc_file = os.path.join(self.data_root, f"cbis_ddsm_{split}_set.xlsx")

        desc_pd = pd.read_excel(pd.ExcelFile(desc_file))

        desc_pd = desc_pd.loc[desc_pd["assessment"].isin(birads)]
        desc_pd = desc_pd.loc[desc_pd["abnormality type"].isin(abnormality)]

        self.lesion_feats_label = desc_pd.columns[21:].to_numpy()
        print("# Samples:", desc_pd.shape)

        self.classes = ["FATTY", "SCATTERED", "HETEROGENEOUSLY_DENSE", "EXTREMELY_DENSE"]

        self.transform = transform

        self.image_paths = []
        self.images = []
        self.labels = []

        self.patients = []
        self.mask_paths = []

        for idx, row in tqdm(desc_pd.iterrows()):
            # if idx > 500:
            #     break

            mamm_file = row["image file path"]
            breast_density = row['breast density']
            abnormality_type = row["abnormality type"]
            patient_id = row["patient_id"]

            # abnormality type, mass shape, mass margin
            # calcification type, calcification distribution
            # lesion_feat = row[21:].to_numpy(dtype=np.int)

            # Image paths
            if abnormality_type == "mass":
                mamm_file_path = os.path.join(mass_image_root, mamm_file)
            elif abnormality_type == "calcification":
                mamm_file_path = os.path.join(calc_image_root, mamm_file)

            p = Path(mamm_file_path)

            mamm_dir = os.path.join(*p.parts[:-1])

            mamm_file_path = os.path.join(mamm_dir, "000000.png")

            # Images
            self.image_paths.append(mamm_file_path)

            # Labels
            if breast_density == 0:
                breast_density = 1
            label = breast_density - 1

            self.labels.append(label)

            self.patients.append(patient_id)

            # Masks
            all_match_records = desc_pd[desc_pd['image file path'] == mamm_file]
            all_masks = []
            for _idx, _row in all_match_records.iterrows():
                mask_file = _row['ROI mask file path']
            

                if abnormality_type == "mass":
                    mask_file_path = os.path.join(mass_image_root, mask_file)
                elif abnormality_type == "calcification":
                    mask_file_path = os.path.join(calc_image_root, mask_file)

                p = Path(mask_file_path)
                mask_dir = os.path.join(*p.parts[:-3])

                mask_file_path = None
                mask_area = float("inf")

                for img_path in glob.glob(os.path.join(mask_dir, "**", "**", "*.png")):
                    img = Image.open(img_path)
                    area = img.size[0] * img.size[1]

                    if (mask_file_path is None) or (area > mask_area):
                        mask_file_path = img_path
                        mask_area = area

                all_masks.append(mask_file_path)
            
            self.mask_paths.append(all_masks)

        max_num_masks = max(len(mask_paths) for mask_paths in self.mask_paths)

        self.mask_paths = [mask_paths + [''] * (max_num_masks - len(mask_paths)) for mask_paths in self.mask_paths]


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        image_path = self.image_paths[index]

        image = Image.open(image_path)
        
        mask_path = self.mask_paths[index]        
        
        label = self.labels[index]

        if self.transform:
            image = self.transform(image)            

        if self.get_mask:
            return dict({"image": image, "label": label, "image_path": image_path, "mask_path": mask_path})
        else:
            return dict({"image": image, "label": label, "image_path": image_path})

    def get_all_images(self):
        return self.image_paths
    
    def get_all_masks(self):
        return self.mask_paths

    def get_all_lesion_feats(self):
        return self.lesion_feats

    def get_all_labels(self):
        return self.labels

    def get_lesion_feats_label(self):
        return self.lesion_feats_label

    def get_all_patients(self):
        return self.patients

    def get_classes(self):
        return self.classes
    

class CBIS_DDSM_whole_mamm_abnormality_dataset(Dataset):
    def __init__(
        self,
        data_root,
        split,
        birads=[0, 1, 2, 3, 4, 5],
        abnormality=["mass", "calcification"],
        data_dir=None, # "CBIS_DDSM_pngs"
        transform=None,
        get_mask=False
    ):
        self.data_root = data_root
        self.get_mask = get_mask

        image_root = os.path.join(self.data_root, data_dir)
        mass_image_root = os.path.join(image_root, "mass", split)
        calc_image_root = os.path.join(image_root, "calc", split)

        desc_file = os.path.join(self.data_root, f"cbis_ddsm_{split}_set.xlsx")

        desc_pd = pd.read_excel(pd.ExcelFile(desc_file))

        desc_pd = desc_pd.loc[desc_pd["assessment"].isin(birads)]
        desc_pd = desc_pd.loc[desc_pd["abnormality type"].isin(abnormality)]

        self.lesion_feats_label = desc_pd.columns[21:].to_numpy()
        print("# Samples:", desc_pd.shape)
        # print('Columns:', self.lesion_feats_label)

        
        self.classes = ["MASS", "CALCIFICATION"]

        self.transform = transform

        self.image_paths = []
        self.images = []
        self.labels = []
        self.patients = []
        self.mask_paths = []

        for idx, row in tqdm(desc_pd.iterrows()):
            # if idx > 500:
            #     break

            mamm_file = row["image file path"]
            abnormality_type = row["abnormality type"]
            patient_id = row["patient_id"]

            # abnormality type, mass shape, mass margin
            # calcification type, calcification distribution
            # lesion_feat = row[21:].to_numpy(dtype=np.int)

            

            # Image paths
            if abnormality_type == "mass":
                mamm_file_path = os.path.join(mass_image_root, mamm_file)
            elif abnormality_type == "calcification":
                mamm_file_path = os.path.join(calc_image_root, mamm_file)

            p = Path(mamm_file_path)

            mamm_dir = os.path.join(*p.parts[:-1])

            mamm_file_path = os.path.join(mamm_dir, "000000.png")

            # Images
            self.image_paths.append(mamm_file_path)

            # Labels
            label = int(abnormality_type == "mass")

            self.labels.append(label)

            self.patients.append(patient_id)

            # Masks
            all_match_records = desc_pd[desc_pd['image file path'] == mamm_file]
            all_masks = []
            for _idx, _row in all_match_records.iterrows():
                mask_file = _row['ROI mask file path']
            

                if abnormality_type == "mass":
                    mask_file_path = os.path.join(mass_image_root, mask_file)
                elif abnormality_type == "calcification":
                    mask_file_path = os.path.join(calc_image_root, mask_file)

                p = Path(mask_file_path)
                mask_dir = os.path.join(*p.parts[:-3])

                mask_file_path = None
                mask_area = float("inf")

                for img_path in glob.glob(os.path.join(mask_dir, "**", "**", "*.png")):
                    img = Image.open(img_path)
                    area = img.size[0] * img.size[1]

                    if (mask_file_path is None) or (area > mask_area):
                        mask_file_path = img_path
                        mask_area = area

                all_masks.append(mask_file_path)
            
            self.mask_paths.append(all_masks)

        max_num_masks = max(len(mask_paths) for mask_paths in self.mask_paths)

        self.mask_paths = [mask_paths + [''] * (max_num_masks - len(mask_paths)) for mask_paths in self.mask_paths]

        # Check label at mammogram-level (For 2 classes case)
        # If a mammogram contain at least one cancerous lesion, then label is 'malignant'.
        # Otherwise, label will be 'benign'
        abnormality_check = defaultdict(set)

        for idx, img_path in enumerate(self.image_paths):
            abnormality_check[img_path].add(self.labels[idx])

        for idx, img_path in enumerate(self.image_paths):
            self.labels[idx] = [0, 0]

            for abn in abnormality_check[img_path]:
                self.labels[idx][abn] = 1


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        image_path = self.image_paths[index]

        image = Image.open(image_path)
        
        mask_path = self.mask_paths[index]        
        
        label = np.array(self.labels[index])

        if self.transform:
            image = self.transform(image)            

        if self.get_mask:
            return dict({"image": image, "label": label, "image_path": image_path, "mask_path": mask_path})
        else:
            return dict({"image": image, "label": label, "image_path": image_path})

    def get_all_images(self):
        return self.image_paths
    
    def get_all_masks(self):
        return self.mask_paths

    def get_all_lesion_feats(self):
        return self.lesion_feats

    def get_all_labels(self):
        return self.labels

    def get_lesion_feats_label(self):
        return self.lesion_feats_label

    def get_all_patients(self):
        return self.patients


class CBIS_DDSM_dataset_tfds(Dataset):
    def __init__(
        self,
        data_root,
        split,
        class_type,
        birads=[0, 1, 2, 3, 4, 5],
        abnormality=["mass", "calcification"],
        official_split=True,
        incl_bg=False,
        incl_tabular=False,
        tabular_methodist_feats_only=False,
        label_type="pathology",
        transform=None,
    ):

        self.data_root = data_root

        if official_split:
            dataset_dir = "CBIS_DDSM_tfds_offical_split"
        else:
            dataset_dir = "CBIS_DDSM_tfds"
        image_root = os.path.join(self.data_root, dataset_dir, split)

        
        self.incl_tabular = incl_tabular
        self.label_type = label_type
        if incl_tabular:
            self.tabular_dataset = CBIS_DDSM_tabular_dataset(
                data_root=self.data_root,
                split="train" if split in ['train', 'val'] else "test",
                birads=birads,
                class_type=class_type,
                abnormality=abnormality,
                methodist_feats_only=tabular_methodist_feats_only,
                label_type=label_type
            )

        if label_type == "pathology":
            if class_type == "2classes":
                self.classes = ["BENIGN", "MALIGNANT"]
            elif class_type == "4classes":
                self.classes = [
                    "BENIGN_CALCIFICATION",
                    "BENIGN_MASS",
                    "MALIGNANT_CALCIFICATION",
                    "MALIGNANT_MASS",
                ]
        else:
            self.classes = self.tabular_dataset.get_classes()

        if incl_bg:
            self.classes.append("BACKGROUND")

        self.transform = transform

        self.image_paths = []
        self.labels = []

        idx = 0
        for img_path in tqdm(glob.glob(
            os.path.join(image_root, "**", "*.png"), recursive=True
        )):
            
          
            class_name = os.path.basename(os.path.dirname(img_path))

            if class_name == "BACKGROUND" and not incl_bg:
                continue

            # Check BIRADS        
            try:
                clinical_label = self.tabular_dataset.get_clinical_label_by_id(os.path.basename(img_path))
                clinical_label = pd.Series(clinical_label).fillna("-1").tolist()
                birad = clinical_label[-1]
            except ValueError:
                continue

            if birad not in birads:
                continue

            # if idx == 500:
            #     break
            # idx += 1

            # Labels
            if class_name == "BACKGROUND":
                self.labels.append(self.classes.index("BACKGROUND"))
            else:
                pathology, abnormality_type = class_name.split("_")

                if abnormality_type.lower() not in abnormality:
                    continue

                if class_type == "2classes":
                    label = int(pathology == "MALIGNANT")
                elif class_type == "4classes":
                    label = self.classes.index(
                        (pathology + "_" + abnormality_type).upper()
                    )

                self.labels.append(label)

            self.image_paths.append(img_path)

        # print("#Samples", len(self.image_paths))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        image_path = self.image_paths[index]

        label = self.labels[index]
        
        image = Image.open(image_path)
        norm_image = cv2.normalize(
            np.array(image), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )
        image = Image.fromarray(norm_image).convert("RGB")

        if self.transform:
            image = self.transform(image)

        patient = self.tabular_dataset.get_patient_by_id(os.path.basename(image_path))
        if self.incl_tabular:
            feat_vec = self.tabular_dataset.get_feature_vector_by_id(os.path.basename(image_path))
            feat_vec_names = self.tabular_dataset.get_feature_vectors_names()
            clinical_label = self.tabular_dataset.get_clinical_label_by_id(os.path.basename(image_path))

            if self.label_type != "pathology":
                label = np.array(self.tabular_dataset.get_label_by_id(os.path.basename(image_path)))

            # TODO: Solved
            # Need to come back later to solve the uncollate problem when there is a NaN in 'clinical_label'
            clinical_label = pd.Series(clinical_label).fillna("-1").tolist()
            breast_side, density_id, abnormal, mass_shape, mass_margin, calc_morph, calc_dist, birad = clinical_label
           
            breast_density_dict = {
                1: "Fatty",
                2: "Scattered",
                3: "Heterogeneously Dense",
                4: "Extremely Dense",
            }

            missing_features = {
                                # EMBED
                                "mass_dense": "-1", \
                                "marital_status": "-1", "ethnicity": "-1", \
                                "ethnic_group": "-1", "age": -1, "findings": "-1"
                            }
            

           
            ret_dict =  dict({   
                            "patient": patient, "image": image, "feat_vec": feat_vec, "feat_vec_names": feat_vec_names, "label": label, \
                            "breast_side": breast_side, "breast_density": breast_density_dict[density_id], "abnormal": abnormal, "mass_shape": mass_shape, \
                            "mass_margin": mass_margin, "calc_morph": calc_morph, "calc_dist": calc_dist, "birad": birad, "image_path": image_path, \
                            "dataset_name": "cbis-ddsm"
                        })
            

            ret_dict.update(missing_features)

            return ret_dict

            # return dict({"image": image, "feat_vec": feat_vec, "label": label, "image_path": image_path})

        return dict({"patient": patient, "image": image, "label": label, "image_path": image_path})

    def get_all_labels(self):
        return self.labels
    
    def get_feature_dim(self):
        if self.incl_tabular:
            return self.tabular_dataset.get_feature_dim()
        return 0

    def get_classes(self):
        return self.classes
    
    def get_all_images(self):
        return self.image_paths
    
    def get_all_patients(self):
        all_patients = []
        for image_path in self.image_paths:
            patient = self.tabular_dataset.get_patient_by_id(os.path.basename(image_path))
            all_patients.append(patient)

        return all_patients


class CBIS_DDSM_tabular_dataset(Dataset):
    def __init__(
        self,
        data_root,
        split,
        class_type,
        birads=[0, 1, 2, 3, 4, 5],
        abnormality=["mass", "calcification"],
        methodist_feats_only=False,
        label_type="pathology",
        transform=None,
    ):
        self.data_root = data_root

        desc_file = os.path.join(self.data_root, f"cbis_ddsm_{split}_set.xlsx")

        desc_pd = pd.read_excel(pd.ExcelFile(desc_file))
        desc_pd.replace({"breast density": 0}, 1, inplace=True)

        # CLINICAL LABELS
        self.clinical_labels = desc_pd[["left or right breast", "breast density", "abnormality type", "mass shape", \
                                        "mass margins", "calc type", "calc distribution", \
                                        "assessment"]].copy()
    
        self.clinical_labels = self.clinical_labels.loc[self.clinical_labels["assessment"].isin(birads)]
        self.clinical_labels = self.clinical_labels.loc[self.clinical_labels["abnormality type"].isin(abnormality)]
        self.clinical_labels = self.clinical_labels.to_numpy()

        # LESION IDS
        self.lesion_ids = desc_pd[['image file path', 'abnormality id', \
                                   'assessment', 'abnormality type']]
        self.lesion_ids = self.lesion_ids.loc[self.lesion_ids["assessment"].isin(birads)]
        self.lesion_ids = self.lesion_ids.loc[self.lesion_ids["abnormality type"].isin(abnormality)]


        # CLINICAL FEATS
        if methodist_feats_only:
            desc_pd = CBIS_DDSM_tabular_dataset.get_methodist_feats_frame(desc_pd)
        else:
            desc_pd = CBIS_DDSM_tabular_dataset.get_original_feats_frame(desc_pd)

        desc_pd = desc_pd.loc[desc_pd["assessment"].isin(birads)]
        desc_pd = desc_pd.loc[desc_pd["abnormality type"].isin(abnormality)]

        # print("# Samples:", desc_pd.shape)

        
        self.lesion_feats = desc_pd.iloc[:, 3:-1].to_numpy()
        self.lesion_feats_names = desc_pd.columns[3:-1].to_list()

        assert self.lesion_feats.shape[0] == self.lesion_ids.shape[0], f"Data split: {split}, {self.lesion_feats.shape[0]} is not equal to {self.lesion_ids.shape[0]}"

        self.label_type = label_type

        if label_type == "pathology":
            self.labels = desc_pd.iloc[:, -1].to_numpy().astype(int)

            if class_type == "2classes":
                self.classes = ["BENIGN", "MALIGNANT"]
            else:
                raise ValueError("Unknown class type.")
        elif methodist_feats_only:
            if label_type == 'mass_appearance':
                self.classes = ['APPEARANCE OVAL', 'APPEARANCE ROUND',
                                'APPEARANCE IRREGULAR', 'APPEARANCE SPICULATED',
                                'APPEARANCE CIRCUMSCRIBED', 'APPEARANCE OBSCURED',
                                'APPEARANCE MICROLOBULATED', 'APPEARANCE INDISTINCT', 
                                'ASYMMETRY/DISTORTION ARCHITECTURAL DISTORTION',
                                'ASYMMETRY/DISTORTION ASYMMETRY']             
            elif label_type == 'calc_morph':
                self.classes = ['CALCIFICATIONS VASCULAR CALCIFICATIONS',
                                'CALCIFICATIONS OTHER CALCIFICATIONS', 'MORPHOLOGY HETEROGENEOUS',
                                'MORPHOLOGY AMORPHOUS', 'MORPHOLOGY ROUND OR PUNCTATE',
                                'MORPHOLOGY LINEAR']
            elif label_type == 'calc_dist':
                self.classes = ['DISTRIBUTION LINEAR', 'DISTRIBUTION SEGMENTAL',
                                'DISTRIBUTION CLUSTERED', 'DISTRIBUTION REGIONAL OR DIFFUSE']
                
            # labels
            self.labels = desc_pd[self.classes].to_numpy().tolist()
        else:
            if label_type == 'mass_shape':
                self.classes = ["mass shape-OVAL", "mass shape-ROUND", "mass shape-IRREGULAR", "mass shape-LOBULATED",
                                "mass shape-ARCHITECTURAL_DISTORTION", "mass shape-ASYMMETRIC_BREAST_TISSUE"] 
            elif label_type == 'mass_margin':
                self.classes = ["mass margin-SPICULATED", "mass margin-CIRCUMSCRIBED", "mass margin-OBSCURED", 
                                "mass margin-MICROLOBULATED", "mass margin-ILL_DEFINED"]            
            elif label_type == 'calc_morph':
                self.classes = ["calc type-VASCULAR", "calc type-DYSTROPHIC", "calc type-EGGSHELL", "calc type-LARGE_RODLIKE",
                                "calc type-LUCENT_CENTER", "calc type-LUCENT_CENTERED", "calc type-MILK_OF_CALCIUM", "calc type-PLEOMORPHIC",
                                "calc type-SKIN", "calc type-COARSE", "calc type-AMORPHOUS", "calc type-PUNCTATE", "calc type-ROUND_AND_REGULAR",
                                "calc type-FINE_LINEAR_BRANCHING"]
            elif label_type == 'calc_dist':
                self.classes = ["calc dist-LINEAR", "calc dist-SEGMENTAL", "calc dist-CLUSTERED", 
                                "calc dist-REGIONAL", "calc dist-DIFFUSELY_SCATTERED"]

                
            # labels
            self.labels = desc_pd[self.classes].to_numpy().tolist()


        self.patients = desc_pd["MRN"].tolist()



        self.transform = transform

    @staticmethod
    def get_original_feats_frame(data_pd):
        result = pd.concat(
            [
                data_pd[["patient_id", "assessment", "abnormality type"]].reset_index(
                    drop=True
                ),
                data_pd[data_pd.columns[16:-1]].reset_index(drop=True),
                (data_pd["pathology"] == "MALIGNANT")
                .astype(int)
                .reset_index(drop=True),
            ],
            axis=1,
        )
        result.rename(
            columns={"patient_id": "MRN", "pathology": "Benign (0)/Malignant (1)"},
            inplace=True,
        )

        return result

    @staticmethod
    def get_methodist_feats_frame(data_pd):
        frame = {
            "MRN": data_pd["patient_id"],
            "assessment": data_pd["assessment"],
            "abnormality type": data_pd["abnormality type"],
            "Left Breast": 1 - data_pd["binarized left or right breast"],
            "Right Breast": data_pd["binarized left or right breast"],
            "MAMMOGRAM DENSITY": data_pd["breast density"].replace(
                {
                    1: "Fatty",
                    2: "Scattered",
                    3: "Heterogeneously Dense",
                    4: "Extremely Dense",
                }
            ),
            "APPEARANCE OVAL": data_pd["mass shape-OVAL"],
            "APPEARANCE ROUND": data_pd["mass shape-ROUND"],
            "APPEARANCE IRREGULAR": data_pd["mass shape-IRREGULAR"],
            "APPEARANCE SPICULATED": data_pd["mass margin-SPICULATED"],
            "APPEARANCE CIRCUMSCRIBED": data_pd["mass margin-CIRCUMSCRIBED"],
            "APPEARANCE OBSCURED": data_pd["mass margin-OBSCURED"],
            "APPEARANCE MICROLOBULATED": data_pd["mass margin-MICROLOBULATED"]
            | data_pd["mass shape-LOBULATED"],
            "APPEARANCE INDISTINCT": data_pd["mass margin-ILL_DEFINED"],
            "CALCIFICATIONS VASCULAR CALCIFICATIONS": data_pd["calc type-VASCULAR"],
            "CALCIFICATIONS OTHER CALCIFICATIONS": data_pd["calc type-DYSTROPHIC"]
            | data_pd["calc type-EGGSHELL"]
            | data_pd["calc type-LARGE_RODLIKE"]
            | data_pd["calc type-LUCENT_CENTER"]
            | data_pd["calc type-LUCENT_CENTERED"]
            | data_pd["calc type-MILK_OF_CALCIUM"]
            | data_pd["calc type-PLEOMORPHIC"]
            | data_pd["calc type-SKIN"],
            "MORPHOLOGY HETEROGENEOUS": data_pd["calc type-COARSE"],
            "MORPHOLOGY AMORPHOUS": data_pd["calc type-AMORPHOUS"],
            "MORPHOLOGY ROUND OR PUNCTATE": data_pd["calc type-PUNCTATE"]
            | data_pd["calc type-ROUND_AND_REGULAR"],
            "MORPHOLOGY LINEAR": data_pd["calc type-FINE_LINEAR_BRANCHING"],
            "DISTRIBUTION LINEAR": data_pd["calc dist-LINEAR"],
            "DISTRIBUTION SEGMENTAL": data_pd["calc dist-SEGMENTAL"],
            "DISTRIBUTION CLUSTERED": data_pd["calc dist-CLUSTERED"],
            "DISTRIBUTION REGIONAL OR DIFFUSE": data_pd["calc dist-REGIONAL"]
            | data_pd["calc dist-DIFFUSELY_SCATTERED"],
            "ASYMMETRY/DISTORTION ARCHITECTURAL DISTORTION": data_pd[
                "mass shape-ARCHITECTURAL_DISTORTION"
            ],
            "ASYMMETRY/DISTORTION ASYMMETRY": data_pd[
                "mass shape-ASYMMETRIC_BREAST_TISSUE"
            ]
            | data_pd["mass shape-FOCAL_ASYMMETRIC_DENSITY"],
            "Benign (0)/Malignant (1)": (data_pd["pathology"] == "MALIGNANT").astype(
                int
            ),
        }
        result = pd.DataFrame(frame)

        transformer = OneHotEncoder()
        transformer.fit(result["MAMMOGRAM DENSITY"].to_numpy().reshape(-1, 1))

        # check if mammogram density has 4 categories
        assert len(transformer.categories_[0]) == 4

        onehot_density = transformer.transform(
            result["MAMMOGRAM DENSITY"].to_numpy().reshape(-1, 1)
        ).toarray()

        result = pd.concat(
            [
                result.reset_index(drop=True),
                pd.DataFrame(
                    onehot_density,
                    columns="MAMMOGRAM DENSITY " + transformer.categories_[0],
                ).reset_index(drop=True),
            ],
            axis=1,
        )

        # rearrange columns
        result = result[
            result.columns[:5].tolist()
            + result.columns[-4:].tolist()
            + result.columns[6:-4].tolist()
        ]

        return result

    def __len__(self):
        return len(self.lesion_feats)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        lesion_feat = self.lesion_feats[index]
        label = self.labels[index]
        clinical_label = self.clinical_labels[index]

        if self.transform:
            lesion_feat = self.transform(lesion_feat)

        return dict({"label": label, "feat_vec": lesion_feat, "clinical_label": clinical_label})

    def get_classes(self):
        _classes = self.classes
        if self.label_type in ['mass_shape', 'mass_margin', 'calc_morph', 'calc_dist']:
            _classes = [_class.split('-')[1] for _class in _classes]
        return _classes

    def get_all_lesion_ids(self):
        return self.lesion_ids

    def get_all_feature_vectors(self):
        return self.lesion_feats
    
    def get_feature_vectors_names(self):
        return self.lesion_feats_names

    def get_feature_dim(self):
        return self.lesion_feats.shape[1]

    def get_all_labels(self):
        return self.labels
    
    def get_all_clinicaL_labels(self):
        return self.clinical_labels

    def get_all_patients(self):
        return self.patients    
    
    def get_satisfied_records_by_id(self, img_filename):
        mamm_id, mask_id, abnorm_id, _ = img_filename.split('#')

        lesion_ids = self.get_all_lesion_ids()

        try:
            finding_idx = ((lesion_ids['image file path'].str.contains(mamm_id)) \
                            & (lesion_ids['image file path'].str.contains(mask_id)) \
                            & (lesion_ids['abnormality id'] == int(abnorm_id.split('_')[1]))).tolist().index(True)
        except:
            raise ValueError(f"Could not find satisfied record for {mamm_id} {mask_id} {abnorm_id}")
        
        return finding_idx

    def get_feature_vector_by_id(self, img_filename):
        finding_idx = self.get_satisfied_records_by_id(img_filename)
        
        return self.lesion_feats[finding_idx]
    
    def get_clinical_label_by_id(self, img_filename):
        finding_idx = self.get_satisfied_records_by_id(img_filename)
        
        return self.clinical_labels[finding_idx]
    
    def get_label_by_id(self, img_filename):
        finding_idx = self.get_satisfied_records_by_id(img_filename)
        
        return self.labels[finding_idx]
    
    def get_patient_by_id(self, img_filename):
        finding_idx = self.get_satisfied_records_by_id(img_filename)
        
        return self.patients[finding_idx]
