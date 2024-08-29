import re
import pathlib
import cv2
import os
import pandas as pd
import numpy as np
import torch
import glob
import multiprocessing
import torchvision.transforms.functional as TF

from torch.utils.data import Dataset
from PIL import Image
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import KBinsDiscretizer
from collections import defaultdict


class EMBED_dataset_tfds(Dataset):
    def __init__(
        self,
        data_root,
        split,
        class_type,
        dataset_dir="images_tfds",
        abnormality=["mass", "calcification"],
        incl_bg=False,
        incl_tabular=False,
        incl_findings=True,
        incl_breast_dense=True,
        incl_breast_side=True,
        incl_findings_feats=True,
        incl_demography=True,
        transform=None,
    ):

        self.data_root = data_root

        image_root = os.path.join(
            self.data_root, "EMBED_images_pngs", "findings", dataset_dir, split
        )

        if class_type == "2classes":
            self.classes = ["BENIGN", "MALIGNANT"]
        elif class_type == "4classes":
            self.classes = [
                "BENIGN_CALCIFICATION",
                "BENIGN_MASS",
                "MALIGNANT_CALCIFICATION",
                "MALIGNANT_MASS",
            ]

        if incl_bg:
            self.classes.append("BACKGROUND")

        self.incl_tabular = incl_tabular
        if incl_tabular:
            self.tabular_dataset = EMBED_tabular_dataset(
                data_root=self.data_root,
                class_type=class_type,
                abnormality=abnormality,
                incl_findings=incl_findings,
                incl_breast_dense=incl_breast_dense,
                incl_breast_side=incl_breast_side,
                incl_findings_feats=incl_findings_feats,
                incl_demography=incl_demography,
            )
            self.feat_vectors_list = []
            self.clinical_labels_list = []
            self.patients = []

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

            if incl_tabular:
                feat_vectors = self.tabular_dataset.get_feature_vectors_by_id(
                    os.path.basename(img_path)
                )

                clinical_label = self.tabular_dataset.get_clinical_label_by_id(
                    os.path.basename(img_path)
                )
                patient = self.tabular_dataset.get_patient_by_id(
                    os.path.basename(img_path)
                )

                for feat_id in range(feat_vectors.shape[0]):
                    if feat_id > 0:
                        self.image_paths.append(img_path)
                        self.labels.append(label)

                    self.feat_vectors_list.append(feat_vectors[feat_id, :])
                    self.clinical_labels_list.append(clinical_label[feat_id, :])
                    self.patients.append(patient[feat_id])

        print("#Samples", len(self.image_paths))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        image_path = self.image_paths[index]
        label = self.labels[index]
        image = Image.open(image_path)

        try:
            norm_image = cv2.normalize(
                np.array(image), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
            )
        except cv2.error:
            norm_image = cv2.normalize(
                np.array(image), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
            )
        except OSError:
            norm_image = cv2.normalize(
                np.array(image), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
            )

        image = Image.fromarray(norm_image).convert("RGB")

        if self.transform:
            image = self.transform(image)

        patient = self.patients[index]
        if self.incl_tabular:
            feat_vec_names = self.tabular_dataset.get_feature_vectors_names()

            feat_vector = self.feat_vectors_list[index]
            clinical_label = self.clinical_labels_list[index]

            clinical_label = pd.Series(clinical_label).fillna("-1").tolist()
            (
                _,
                _,
                _,
                _,
                tissueden,
                side,
                massshape,
                massmargin,
                massdens,
                calcfind,
                calcdistri,
                marital,
                ethnicity,
                ethnic_group,
                age,
                birad,
                findings,
            ) = clinical_label

            return dict(
                {
                    "patient": patient,
                    "image": image,
                    "feat_vec": feat_vector,
                    "feat_vec_names": feat_vec_names,
                    "label": label,
                    "tissueden": tissueden,
                    "breast_side": side,
                    "mass_shape": massshape,
                    "mass_margin": massmargin,
                    "mass_dense": massdens,
                    "calc_morph": calcfind,
                    "calc_dist": calcdistri,
                    "marital_status": marital,
                    "ethnicity": ethnicity,
                    "ethnic_group": ethnic_group,
                    "age": age,
                    "findings": findings,
                    "birad": birad,
                    "image_path": image_path,
                    "dataset_name": "embed",
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


class EMBED_tabular_dataset(Dataset):
    def __init__(
        self,
        data_root,
        class_type,
        abnormality=None,  # None if you do not want to use 'abnormality' to filter
        incl_findings=True,
        incl_breast_dense=True,
        incl_breast_side=True,
        incl_findings_feats=True,
        incl_demography=True,
        transform=None,
    ):
        self.data_root = data_root

        desc_pd = self.get_cancer_roi_df()

        self.desc_pd = desc_pd.copy()

        # remove records without biopsies
        self.desc_pd.dropna(subset=["path_severity"], inplace=True)

        # rename BI-RADS
        asses_dict = {
            "A": 0,
            "N": 1,
            "B": 2,
            "P": 3,
            "S": 4,
            "M": 5,
            "K": 6,
            "X": -1,  # No Assessment
        }
        desc_pd["asses"] = desc_pd["asses"].replace(asses_dict)

        # Fill nan entry in column Age with mean age.
        desc_pd["age_at_study"] = desc_pd["age_at_study"].fillna(
            desc_pd["age_at_study"].mean()
        )

        if abnormality is not None:
            if len(abnormality) == 1:
                if "mass" in abnormality:
                    satisfied_rows = (
                        (desc_pd["mass"] == 1)
                        | (desc_pd["asymmetry"] == 1)
                        | (desc_pd["arch_distortion"] == 1)
                    )
                elif "calcification" in abnormality:
                    satisfied_rows = desc_pd["calc"] == 1
            elif (
                len(abnormality) == 2
                and ("mass" in abnormality)
                and ("calcification" in abnormality)
            ):
                satisfied_rows = (
                    (desc_pd["mass"] == 1)
                    | (desc_pd["asymmetry"] == 1)
                    | (desc_pd["arch_distortion"] == 1)
                    | (desc_pd["calc"] == 1)
                )

            desc_pd = desc_pd.loc[satisfied_rows]

        self.clinical_labels = self.get_clinical_labels(desc_pd)
        self.clinical_labels["asses"] = self.clinical_labels["asses"].replace(
            asses_dict
        )
        self.clinical_labels = self.clinical_labels.to_numpy()

        self.lesion_ids = desc_pd[["anon_dicom_path", "ROI_separated"]]

        self.incl_findings = incl_findings
        self.incl_breast_dense = incl_breast_dense
        self.incl_breast_side = incl_breast_side
        self.incl_findings_feats = incl_findings_feats
        self.incl_demography = incl_demography

        desc_pd = self.get_original_feats_frame(desc_pd)

        print("# Samples:", desc_pd.shape)
        print("# Lesions ID:", self.lesion_ids.shape)

        assert desc_pd.shape[0] == self.lesion_ids.shape[0]
        assert desc_pd.shape[0] == self.clinical_labels.shape[0]

        self.lesion_feats = desc_pd.iloc[:, 2:-1].to_numpy()
        self.lesion_feats_names = desc_pd.columns[2:-1].to_list()
        self.labels = desc_pd.iloc[:, -1].to_numpy().astype(int)
        self.patients = desc_pd["empi_acc_anon"].tolist()

        if class_type == "2classes":
            self.classes = ["BENIGN", "MALIGNANT"]
        else:
            raise ValueError("Unknown class type.")

        self.transform = transform

    def get_clinical_labels(self, desc_pd):
        breast_side_dict = {
            "L": "Left breast",
            "R": "Right breast",
            "B": "Both breasts",
        }

        tissueden_dict = {
            1: "Fatty",
            2: "Scattered",
            3: "Heterogeneously dense",
            4: "Extremely Dense",
            5: "Normal Male",
        }

        massshape_dict = {
            "G": "Generic",
            "R": "Round",
            "O": "Oval",
            "X": "Irregular",
            "Q": "Questioned architectural distortion",
            "A": "Architectural distortion",
            "T": "Asymmetric tubular structure/solitary dilated duct",
            "N": "Intramammary lymph nodes",
            "B": "Global asymmetry",
            "F": "Focal asymmetry",
            "V": "Developing asymmetry",
            "Y": "Lymph node",
            "S": "Asymmetry",
        }

        massmargin_dict = {
            "D": "Circumscribed",
            "U": "Obscured",
            "M": "Microlobulated",
            "I": "Indistinct",
            "S": "Spiculated",
        }

        massdens_dict = {
            "+": "High density",
            "=": "Isodense",
            "-": "Low density",
            "0": "Fat containing",
        }

        calcfind_dict = {  # be careful with feature '9': Benign
            "A": "Amorphous",
            "9": "Benign",
            "H": "Coarse heterogenous",
            "C": "Coarse popcornlike",
            "D": "Dystrophic",
            "E": "Rim",
            "F": "Fine-linear",
            "B": "Fine linear-branching",
            "G": "Generic",
            "I": "Fine pleomorphic",
            "L": "Large rodlike",
            "M": "Milk of calcium",
            "J": "Oil cyst",
            "K": "Pleomorphic",
            "P": "Punctate",
            "R": "Round",
            "S": "Skin",
            "O": "Lucent centered",
            "U": "Suture",
            "V": "Vascular",
            "Q": "Coarse",
        }

        calcdistri_dict = {
            "G": "Grouped",
            "S": "Segmental",
            "R": "Regional",
            "D": "Diffuse/scattered",
            "L": "Linear",
            "C": "Clustered",
        }

        clinical_labels = desc_pd[
            [
                "mass",
                "asymmetry",
                "arch_distortion",
                "calc",
                "tissueden",
                "side",
                "massshape",
                "massmargin",
                "massdens",
                "calcfind",
                "calcdistri",
                "MARITAL_STATUS_DESC",
                "ETHNICITY_DESC",
                "ETHNIC_GROUP_DESC",
                "age_at_study",
                "asses",
            ]
        ].copy()

        clinical_labels = clinical_labels.loc[:, ~clinical_labels.columns.duplicated()]

        clinical_labels.replace({"side": breast_side_dict}, inplace=True)
        clinical_labels.replace({"tissueden": tissueden_dict}, inplace=True)
        clinical_labels.replace({"massshape": massshape_dict}, inplace=True)
        clinical_labels.replace({"massmargin": massmargin_dict}, inplace=True)
        clinical_labels.replace({"massdens": massdens_dict}, inplace=True)
        clinical_labels.replace({"calcfind": calcfind_dict}, inplace=True, regex=True)
        clinical_labels.replace({"calcdistri": calcdistri_dict}, inplace=True)

        clinical_labels.replace({"mass": {0: float("nan"), 1: "mass"}}, inplace=True)
        clinical_labels.replace(
            {"asymmetry": {0: float("nan"), 1: "asymmetry"}}, inplace=True
        )
        clinical_labels.replace(
            {"arch_distortion": {0: float("nan"), 1: "architectural distortion"}},
            inplace=True,
        )
        clinical_labels.replace(
            {"calc": {0: float("nan"), 1: "calcification"}}, inplace=True
        )

        clinical_labels["findings"] = clinical_labels[
            ["mass", "asymmetry", "arch_distortion", "calc"]
        ].apply(lambda x: ", ".join(x.dropna().astype(str)), axis=1)
        return clinical_labels

    def get_original_feats_frame(self, clinical_pd):

        # Binarize 'side'
        clinical_pd["side"] = clinical_pd["side"].replace({"B": "L,R"})
        clinical_pd["side"].value_counts()

        side_mlb = MultiLabelBinarizer()

        side_multilabel = list(
            map(
                lambda x: x.split(",") if isinstance(x, str) else [float("nan")],
                clinical_pd["side"],
            )
        )

        side_mlb.fit(["L", "R"])

        multilabel_bin_side = side_mlb.transform(side_multilabel).tolist()

        assert np.sum(multilabel_bin_side) != 0

        # Binarize 'tissueden'
        tissueden_dict = {
            1: "Fatty",
            2: "Scattered",
            3: "Heterogeneously dense",
            4: "Extremely Dense",
            5: "Normal Male",
        }
        tissueden_transformer = OneHotEncoder(handle_unknown="ignore")
        tissueden_transformer.fit(np.array([1, 2, 3, 4, 5]).reshape(-1, 1))

        onehot_tissueden = tissueden_transformer.transform(
            clinical_pd["tissueden"].to_numpy().reshape(-1, 1)
        ).toarray()
        assert np.sum(onehot_tissueden) != 0

        # Binarize 'massshape'
        massshape_dict = {
            "G": "Generic (G)",
            "R": "Round (R)",
            "O": "Oval (O)",
            "X": "Irregular (X)",
            "Q": "Questioned architectural distortion (Q)",
            "A": "Architectural distortion (A)",
            "T": "Asymmetric tubular structure/solitary dilated duct (T)",
            "N": "Intramammary lymph nodes (N)",
            "B": "Global asymmetry (B)",
            "F": "Focal asymmetry (F)",
            "V": "Developing asymmetry (V)",
            "Y": "Lymph node (Y)",
            "S": "Asymmetry (S)",
        }

        massshape_transformer = OneHotEncoder(handle_unknown="ignore")
        massshape_transformer.fit(
            np.array(
                ["G", "R", "O", "X", "Q", "A", "T", "N", "B", "F", "V", "Y", "S"]
            ).reshape(-1, 1)
        )

        onehot_massshape = massshape_transformer.transform(
            clinical_pd["massshape"].to_numpy().reshape(-1, 1)
        ).toarray()
        assert np.sum(onehot_massshape) != 0

        # Binarize 'massmargin'
        massmargin_dict = {
            "D": "Circumscribed (D)",
            "U": "Obscured (U)",
            "M": "Microlobulated (M)",
            "I": "Indistinct (I)",
            "S": "Spiculated (S)",
        }

        massmargin_transformer = OneHotEncoder(handle_unknown="ignore")
        massmargin_transformer.fit(np.array(["D", "U", "M", "I", "S"]).reshape(-1, 1))

        onehot_massmargin = massmargin_transformer.transform(
            clinical_pd["massmargin"].to_numpy().reshape(-1, 1)
        ).toarray()
        assert np.sum(onehot_massmargin) != 0

        # Binarize 'massdens'
        massdens_dict = {
            "+": "High density",
            "=": "Isodense",
            "-": "Low density",
            "0": "Fat containing",
        }

        massdens_transformer = OneHotEncoder(handle_unknown="ignore")
        massdens_transformer.fit(np.array(["+", "=", "-", "0"]).reshape(-1, 1))

        onehot_massdens = massdens_transformer.transform(
            clinical_pd["massdens"].to_numpy().reshape(-1, 1)
        ).toarray()
        assert np.sum(onehot_massdens) != 0

        # Binarize 'calcfind'
        calcfind_dict = {  # be careful with feature '9': Benign
            "A": "Amorphous (A)",
            "9": "Benign (9)",
            "H": "Coarse heterogenous (H)",
            "C": "Coarse popcornlike (C)",
            "D": "Dystrophic (D)",
            "E": "Rim (E)",
            "F": "Fine-linear (F)",
            "B": "Fine linear-branching (B)",
            "G": "Generic (G)",
            "I": "Fine pleomorphic (I)",
            "L": "Large rodlike (L)",
            "M": "Milk of calcium (M)",
            "J": "Oil cyst (J)",
            "K": "Pleomorphic (K)",
            "P": "Punctate (P)",
            "R": "Round (R)",
            "S": "Skin (S)",
            "O": "Lucent centered (O)",
            "U": "Suture (U)",
            "V": "Vascular (V)",
            "Q": "Coarse (Q)",
        }

        calcfind_mlb = MultiLabelBinarizer()

        calcfind_multilabel = list(
            map(
                lambda x: x.split(",") if isinstance(x, str) else [float("nan")],
                clinical_pd["calcfind"],
            )
        )

        calcfind_mlb.fit(
            [
                "A",
                "9",
                "H",
                "C",
                "D",
                "E",
                "F",
                "B",
                "G",
                "I",
                "L",
                "M",
                "J",
                "K",
                "P",
                "R",
                "S",
                "O",
                "U",
                "V",
                "Q",
            ]
        )

        multilabel_bin_calcfind = calcfind_mlb.transform(calcfind_multilabel).tolist()

        assert np.sum(multilabel_bin_calcfind) != 0

        # Binarize 'calcdistri'
        calcdistri_dict = {
            "G": "Grouped (G)",
            "S": "Segmental (S)",
            "R": "Regional (R)",
            "D": "Diffuse/scattered (D)",
            "L": "Linear (L)",
            "C": "Clustered (C)",
        }

        calcdistri_transformer = OneHotEncoder(handle_unknown="ignore")
        calcdistri_transformer.fit(
            np.array(["G", "S", "R", "D", "L", "C"]).reshape(-1, 1)
        )

        onehot_calcdistri = calcdistri_transformer.transform(
            clinical_pd["calcdistri"].to_numpy().reshape(-1, 1)
        ).toarray()
        assert np.sum(onehot_calcdistri) != 0

        # create 4 columns "mass", "asymmetry", "arc"
        mass_list = [0] * clinical_pd.shape[0]
        asymmetry_list = [0] * clinical_pd.shape[0]
        arch_distortion_list = [0] * clinical_pd.shape[0]
        calc_list = [0] * clinical_pd.shape[0]

        # iterate through rows and assign values to the lists based on above rules
        for ind, (_, row) in enumerate(clinical_pd.iterrows()):
            if (
                (row["massshape"] in ["G", "R", "O", "X", "N", "Y", "D", "L"])
                or (row["massmargin"] in ["D", "U", "M", "I", "S"])
                or (row["massdens"] in ["+", "-", "="])
            ):
                mass_list[ind] = 1

            if row["massshape"] in ["T", "B", "S", "F", "V"]:
                asymmetry_list[ind] = 1

            if row["massshape"] in ["Q", "A"]:
                arch_distortion_list[ind] = 1

            if (
                (row["calcdistri"] is not np.nan)
                or (row["calcfind"] is not np.nan)
                or (row["calcnumber"] != 0)
            ):
                calc_list[ind] = 1

        # Append the final image findings columns to the dataframe
        clinical_pd["mass"] = mass_list
        clinical_pd["asymmetry"] = asymmetry_list
        clinical_pd["arch_distortion"] = arch_distortion_list
        clinical_pd["calc"] = calc_list

        # Binarize Marital status
        marital_transformer = OneHotEncoder(handle_unknown="ignore")
        marital_transformer.fit(
            np.array(
                [
                    "Married",
                    "Single",
                    "Divorced",
                    "Widow(er)",
                    "Separated",
                    "Life Partner",
                    "Parted",
                    "Common Law",
                    # "Not Recorded",
                    # "Unknown",
                ]
            ).reshape(-1, 1)
        )

        onehot_marital = marital_transformer.transform(
            clinical_pd["MARITAL_STATUS_DESC"].to_numpy().reshape(-1, 1)
        ).toarray()
        assert np.sum(onehot_marital) != 0

        # Binarize Ethnicity
        ethnicity_transformer = OneHotEncoder(handle_unknown="ignore")
        ethnicity_transformer.fit(
            np.array(
                [
                    "African American  or Black",
                    "Caucasian or White",
                    "Asian",
                    "Native Hawaiian or Other Pacific Islander",
                    "Multiple",
                    "American Indian or Alaskan Native",
                    # "Unknown, Unavailable or Unreported",
                    # "Not Recorded",
                    # "Patient Declines",
                ]
            ).reshape(-1, 1)
        )

        onehot_ethnicity = ethnicity_transformer.transform(
            clinical_pd["ETHNICITY_DESC"].to_numpy().reshape(-1, 1)
        ).toarray()
        assert np.sum(onehot_ethnicity) != 0

        # Binarize Ethnic Group
        ethnicgroup_transformer = OneHotEncoder(handle_unknown="ignore")
        ethnicgroup_transformer.fit(
            np.array(
                [
                    "Non-Hispanic or Latino",
                    "Hispanic or Latino",
                    # "Unreported, Unknown, Unavailable",
                    # "Not Recorded",
                    # "Unknown~Non-Hispanic",
                ]
            ).reshape(-1, 1)
        )

        onehot_ethnicgroup = ethnicgroup_transformer.transform(
            clinical_pd["ETHNIC_GROUP_DESC"].to_numpy().reshape(-1, 1)
        ).toarray()
        assert np.sum(onehot_ethnicgroup) != 0

        # Binarize Age
        age_est = KBinsDiscretizer(n_bins=8, encode="onehot", strategy="uniform")
        age_est.fit(clinical_pd["age_at_study"].to_numpy().reshape(-1, 1))
        onehot_age = age_est.transform(
            clinical_pd["age_at_study"].to_numpy().reshape(-1, 1)
        )
        assert np.sum(onehot_age) != 0

        result_list = [
            pd.DataFrame(
                clinical_pd[["empi_anon", "acc_anon"]]
                .astype(str)
                .apply("-".join, axis=1),
                columns=["empi_acc_anon"],
            ).reset_index(drop=True),
            clinical_pd[["asses"]].reset_index(drop=True),
        ]

        if self.incl_findings:
            result_list.append(
                clinical_pd[
                    ["mass", "asymmetry", "arch_distortion", "calc"]
                ].reset_index(drop=True)
            )

        if self.incl_breast_dense:
            result_list.append(
                pd.DataFrame(
                    onehot_tissueden,
                    columns="tissueden "
                    + tissueden_transformer.categories_[0].astype(str).astype(object),
                ).reset_index(drop=True)
            )

        if self.incl_breast_side:
            result_list.append(
                pd.DataFrame(
                    multilabel_bin_side, columns="side " + side_mlb.classes_
                ).reset_index(drop=True)
            )

        if self.incl_findings_feats:
            result_list.extend(
                [
                    pd.DataFrame(
                        onehot_massshape,
                        columns="massshape "
                        + massshape_transformer.categories_[0]
                        .astype(str)
                        .astype(object),
                    ).reset_index(drop=True),
                    pd.DataFrame(
                        onehot_massmargin,
                        columns="massmargin "
                        + massmargin_transformer.categories_[0]
                        .astype(str)
                        .astype(object),
                    ).reset_index(drop=True),
                    pd.DataFrame(
                        onehot_massdens,
                        columns="massdens "
                        + massdens_transformer.categories_[0]
                        .astype(str)
                        .astype(object),
                    ).reset_index(drop=True),
                    pd.DataFrame(
                        multilabel_bin_calcfind,
                        columns="calcfind " + calcfind_mlb.classes_,
                    ).reset_index(drop=True),
                    pd.DataFrame(
                        onehot_calcdistri,
                        columns="calcdistri "
                        + calcdistri_transformer.categories_[0]
                        .astype(str)
                        .astype(object),
                    ).reset_index(drop=True),
                ]
            )

        if self.incl_demography:
            result_list.extend(
                [
                    pd.DataFrame(
                        onehot_marital,
                        columns="marital "
                        + marital_transformer.categories_[0].astype(str).astype(object),
                    ).reset_index(drop=True),
                    pd.DataFrame(
                        onehot_ethnicity,
                        columns="ethnicity "
                        + ethnicity_transformer.categories_[0]
                        .astype(str)
                        .astype(object),
                    ).reset_index(drop=True),
                    pd.DataFrame(
                        onehot_ethnicgroup,
                        columns="ethnic group "
                        + ethnicgroup_transformer.categories_[0]
                        .astype(str)
                        .astype(object),
                    ).reset_index(drop=True),
                    pd.DataFrame(
                        onehot_age.todense(),
                        columns="VISIT_AGE "
                        + np.arange(age_est.n_bins_[0]).astype(str).astype(object),
                    ).reset_index(drop=True),
                ]
            )

        # Biopsies GT
        result_list.append(
            pd.DataFrame(clinical_pd["path_severity"] <= 1)
            .astype(int)
            .reset_index(drop=True)
        )

        result = pd.concat(result_list, axis=1)

        return result

    def get_cancer_roi_df(self):
        df_clinical = pd.read_csv(
            os.path.join(self.data_root, "tables/EMBED_OpenData_clinical.csv")
        )

        df_clinical = df_clinical[
            [
                "empi_anon",
                "acc_anon",
                "study_date_anon",
                "asses",
                "tissueden",
                "desc",
                "side",
                "path_severity",
                "numfind",
                "total_L_find",
                "total_R_find",
                "massshape",
                "massmargin",
                "massdens",
                "calcfind",
                "calcdistri",
                "calcnumber",
                "ETHNICITY_DESC",
                "ETHNIC_GROUP_DESC",
                "age_at_study",
                "ETHNIC_GROUP_DESC",
                "MARITAL_STATUS_DESC",
            ]
        ]

        # Load metadata and filter for fields needed for the tasks we are showcasing in this notebook
        df_metadata = pd.read_csv(
            os.path.join(self.data_root, "tables/EMBED_OpenData_metadata.csv")
        )
        df_metadata = df_metadata[
            [
                "anon_dicom_path",
                "empi_anon",
                "acc_anon",
                "study_date_anon",
                "StudyDescription",
                "SeriesDescription",
                "FinalImageType",
                "ImageLateralityFinal",
                "ViewPosition",
                "spot_mag",
                "ROI_coords",
                "num_roi",
            ]
        ]

        ############################################################################################
        # Adding columns "mass", "asymmetry", "arch_distortion" and "calc" as a summary of imaging findings contained in the
        # other columns. This will be coded as 1 = present; 0 = absent

        df_findings_count = df_clinical.copy()
        df_findings_count = df_findings_count.reset_index(drop=True)

        # Instantiate lists for the four finding type -  mass, asymmetry, architectural distortion and calcification
        # Default value set to 0.
        mass_list = [0] * df_findings_count.shape[0]
        asymmetry_list = [0] * df_findings_count.shape[0]
        arch_destortion_list = [0] * df_findings_count.shape[0]
        calc_list = [0] * df_findings_count.shape[0]

        # Architectural Distortion is defined as: 'massshape' ['Q', 'A']
        # Asymmetry is defined as: 'massshape' in ['T', 'B', 'S', 'F', 'V']
        # Mass is defined as: 'massshape' in ['G', 'R', 'O', 'X', 'N', 'Y', 'D', 'L']
        #       or 'massmargin' in ['D', 'U', 'M', 'I', 'S']
        #       or 'massdens' in ['+', '-', '=']
        # Calcification: defined as presence of any non-zero or non-null value in "calcdistri", "calcfind" or "calcnumber"

        # iterate through rows and assign values to the lists based on above rules
        for ind, row in df_findings_count.iterrows():
            if (
                (row["massshape"] in ["G", "R", "O", "X", "N", "Y", "D", "L"])
                or (row["massmargin"] in ["D", "U", "M", "I", "S"])
                or (row["massdens"] in ["+", "-", "="])
            ):
                mass_list[ind] = 1

            if row["massshape"] in ["T", "B", "S", "F", "V"]:
                asymmetry_list[ind] = 1

            if row["massshape"] in ["Q", "A"]:
                arch_destortion_list[ind] = 1

            if (
                (row["calcdistri"] is not np.nan)
                or (row["calcfind"] is not np.nan)
                or (row["calcnumber"] != 0)
            ):
                calc_list[ind] = 1

        # Append the final image findings columns to the dataframe
        df_findings_count["mass"] = mass_list
        df_findings_count["asymmetry"] = asymmetry_list
        df_findings_count["arch_distortion"] = arch_destortion_list
        df_findings_count["calc"] = calc_list

        df_clinical = df_findings_count.copy()

        ############################################################################################

        ############################################################################################

        df_merge = pd.merge(df_metadata, df_clinical, on=["acc_anon"])  # All findings
        df_merge.rename(columns={"empi_anon_y": "empi_anon"}, inplace=True)

        # The 'side' column in the clinical data represents the laterality of the finding in that row, and can be L (left),
        # R (right), B (bilateral), or NaN (when there is no finding). Therefore when merging clinical and metadata, we must
        # first match by exam ID and then match the laterality of the clinical finding (side) to the laterality of the image
        # (ImageLateralityFinal). Side "B" and "NaN" can be matched to ImageLateralityFinal both "L" and "R"
        df_merge = df_merge.loc[
            (df_merge.side == df_merge.ImageLateralityFinal)
            | (df_merge.side == "B")
            | (pd.isna(df_merge.side))
        ]

        df_merge.drop_duplicates(inplace=True)

        # Note the significant drop in number of rows after forcing the above laterality match while maintaining the number
        # of patients and exams. A few exams/patients are lost if there is no image to match the side of the finding, which
        # may be due to data entry error or data loss during extraction

        ############################################################################################
        # Filter for 2D and C-view images only. Currently the EMBED AWS Open dataset does not contain any other image types,
        # but we retain this code for future use when it will contain DBT, MRI, and US

        df_merge_2d_cview = df_merge.loc[df_merge.FinalImageType.isin(["2D", "cview"])]

        ############################################################################################
        # we will now get the list of ROIs for each of these images. ROIs are structured as a list of lists, and each image
        # can have 0 to multiple ROIs. We will therefore parse the ROI list to expand it such that each row will contain one ROI.
        # If an image has multiple ROIs, this will result in multiple rows for that image in the resultant dataframe

        # define function
        def separate_roi(df):
            df_list = []
            for ind, row in df.iterrows():
                path = row["anon_dicom_path"]
                roi_num = [int(s) for s in re.findall(r"\b\d+\b", row["ROI_coords"])]
                if len(roi_num) == 4:
                    df_list.append([path, row["ROI_coords"], row["ROI_coords"]])
                else:
                    count = 0
                    roi = []
                    for i in roi_num:
                        count += 1
                        roi.append(i)
                        if count % 4 == 0:
                            df_list.append(
                                [
                                    path,
                                    row["ROI_coords"],
                                    "(("
                                    + str(roi[0])
                                    + ", "
                                    + str(roi[1])
                                    + ", "
                                    + str(roi[2])
                                    + ", "
                                    + str(roi[3])
                                    + "),)",
                                ]
                            )
                            roi = []
            df_roi_sep = pd.DataFrame(df_list)
            df_roi_sep.columns = ["anon_dicom_path", "ROI_coords", "ROI_separated"]
            df_cp = df.copy()
            df_cp = df_cp.merge(
                df_roi_sep.copy(), how="left", on=["anon_dicom_path", "ROI_coords"]
            )
            return df_cp

        ############################################################################################
        # To export ROIs, filter the ones with ROIs
        df_cancer_ROI = df_merge_2d_cview.loc[df_merge_2d_cview.ROI_coords != "()"]

        # Separate multiple ROIs into individual rows
        df_cancer_ROI = separate_roi(df_cancer_ROI)

        # Export to csv
        df_cancer_ROI.drop_duplicates(inplace=True)

        return df_cancer_ROI

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

        # return dict({"label": label, "lesion_feat": lesion_feat})
        return dict(
            {"label": label, "feat_vec": lesion_feat, "clinical_label": clinical_label}
        )

    def get_all_lesion_ids(self):
        return self.lesion_ids

    def get_tabular_df(self):
        return self.desc_pd

    def get_all_feature_vectors(self):
        return self.lesion_feats

    def get_feature_vectors_names(self):
        return self.lesion_feats_names

    def get_feature_dim(self):
        return self.lesion_feats.shape[1]

    def get_all_labels(self):
        return self.labels

    def get_all_patients(self):
        return self.patients

    def get_satisfied_records_by_id(self, img_filename):
        mamm_id, _, bbox = img_filename.split("#")[0].split("_")

        lesion_ids = self.get_all_lesion_ids()

        try:
            finding_ids = (
                (lesion_ids["anon_dicom_path"].str.contains(mamm_id))
                & (
                    lesion_ids["ROI_separated"]
                    .astype(str)
                    .str.contains(re.escape(bbox))
                )
            ).to_numpy()

        except:
            raise ValueError(f"Could not find tabular feature for {mamm_id} {bbox}")

        return finding_ids

    def get_feature_vectors_by_id(self, img_filename):
        finding_ids = self.get_satisfied_records_by_id(img_filename)

        return self.lesion_feats[finding_ids]

    def get_clinical_label_by_id(self, img_filename):
        finding_ids = self.get_satisfied_records_by_id(img_filename)

        return self.clinical_labels[finding_ids]

    def get_label_by_id(self, img_filename):
        finding_ids = self.get_satisfied_records_by_id(img_filename)

        return self.labels[finding_ids]

    def get_patient_by_id(self, img_filename):
        finding_ids = self.get_satisfied_records_by_id(img_filename)

        return np.array(self.patients)[finding_ids]
