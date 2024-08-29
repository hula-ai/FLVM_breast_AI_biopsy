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
                split="train" if split in ["train", "val"] else "test",
                class_type=class_type,
                abnormality=abnormality,
                label_type=label_type,
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
        for img_path in glob.glob(
            os.path.join(image_root, "**", "*.png"), recursive=True
        ):
            class_name = os.path.basename(os.path.dirname(img_path))

            if class_name == "BACKGROUND" and not incl_bg:
                continue

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
            feat_vec = self.tabular_dataset.get_feature_vector_by_id(
                os.path.basename(image_path)
            )
            feat_vec_names = self.tabular_dataset.get_feature_vectors_names()
            clinical_label = self.tabular_dataset.get_clinical_label_by_id(
                os.path.basename(image_path)
            )

            if self.label_type != "pathology":
                label = np.array(
                    self.tabular_dataset.get_label_by_id(os.path.basename(image_path))
                )

            clinical_label = pd.Series(clinical_label).fillna("-1").tolist()
            (
                breast_side,
                density_id,
                abnormal,
                mass_shape,
                mass_margin,
                calc_morph,
                calc_dist,
                birad,
            ) = clinical_label

            breast_density_dict = {
                1: "Fatty",
                2: "Scattered",
                3: "Heterogeneously Dense",
                4: "Extremely Dense",
            }

            return dict(
                {
                    "patient": patient,
                    "image": image,
                    "feat_vec": feat_vec,
                    "feat_vec_names": feat_vec_names,
                    "label": label,
                    "breast_side": breast_side,
                    "breast_density": breast_density_dict[density_id],
                    "abnormal": abnormal,
                    "mass_shape": mass_shape,
                    "mass_margin": mass_margin,
                    "calc_morph": calc_morph,
                    "calc_dist": calc_dist,
                    "birad": birad,
                    "image_path": image_path,
                    "dataset_name": "cbis-ddsm",
                }
            )

        return dict(
            {
                "patient": patient,
                "image": image,
                "label": label,
                "image_path": image_path,
            }
        )

    def get_all_labels(self):
        return self.labels

    def get_feature_dim(self):
        if self.incl_tabular:
            return self.tabular_dataset.get_feature_dim()
        return 0

    def get_classes(self):
        return self.classes


class CBIS_DDSM_tabular_dataset(Dataset):
    def __init__(
        self,
        data_root,
        split,
        class_type,
        birads=[0, 1, 2, 3, 4, 5],
        abnormality=["mass", "calcification"],
        label_type="pathology",
        transform=None,
    ):
        self.data_root = data_root

        desc_file = os.path.join(self.data_root, f"cbis_ddsm_{split}_set.xlsx")

        desc_pd = pd.read_excel(pd.ExcelFile(desc_file))
        desc_pd.replace({"breast density": 0}, 1, inplace=True)

        # CLINICAL LABELS
        self.clinical_labels = desc_pd[
            [
                "left or right breast",
                "breast density",
                "abnormality type",
                "mass shape",
                "mass margins",
                "calc type",
                "calc distribution",
                "assessment",
            ]
        ].copy()

        self.clinical_labels = self.clinical_labels.loc[
            self.clinical_labels["assessment"].isin(birads)
        ]
        self.clinical_labels = self.clinical_labels.loc[
            self.clinical_labels["abnormality type"].isin(abnormality)
        ]
        self.clinical_labels = self.clinical_labels.to_numpy()

        # LESION IDS
        self.lesion_ids = desc_pd[
            ["image file path", "abnormality id", "assessment", "abnormality type"]
        ]
        self.lesion_ids = self.lesion_ids.loc[
            self.lesion_ids["assessment"].isin(birads)
        ]
        self.lesion_ids = self.lesion_ids.loc[
            self.lesion_ids["abnormality type"].isin(abnormality)
        ]

        # CLINICAL FEATS
        desc_pd = CBIS_DDSM_tabular_dataset.get_original_feats_frame(desc_pd)

        desc_pd = desc_pd.loc[desc_pd["assessment"].isin(birads)]
        desc_pd = desc_pd.loc[desc_pd["abnormality type"].isin(abnormality)]

        # print("# Samples:", desc_pd.shape)

        self.lesion_feats = desc_pd.iloc[:, 3:-1].to_numpy()
        self.lesion_feats_names = desc_pd.columns[3:-1].to_list()

        assert (
            self.lesion_feats.shape[0] == self.lesion_ids.shape[0]
        ), f"Data split: {split}, {self.lesion_feats.shape[0]} is not equal to {self.lesion_ids.shape[0]}"

        self.label_type = label_type

        # Label Type: Pathology
        self.labels = desc_pd.iloc[:, -1].to_numpy().astype(int)

        if class_type == "2classes":
            self.classes = ["BENIGN", "MALIGNANT"]
        else:
            raise ValueError("Unknown class type.")

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

        return dict(
            {"label": label, "feat_vec": lesion_feat, "clinical_label": clinical_label}
        )

    def get_classes(self):
        _classes = self.classes
        if self.label_type in ["mass_shape", "mass_margin", "calc_morph", "calc_dist"]:
            _classes = [_class.split("-")[1] for _class in _classes]
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
        mamm_id, mask_id, abnorm_id, _ = img_filename.split("#")

        lesion_ids = self.get_all_lesion_ids()

        try:
            finding_idx = (
                (
                    (lesion_ids["image file path"].str.contains(mamm_id))
                    & (lesion_ids["image file path"].str.contains(mask_id))
                    & (lesion_ids["abnormality id"] == int(abnorm_id.split("_")[1]))
                )
                .tolist()
                .index(True)
            )
        except:
            raise ValueError(
                f"Could not find satisfied record for {mamm_id} {mask_id} {abnorm_id}"
            )

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
