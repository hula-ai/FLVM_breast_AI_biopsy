from MultiMEDal_multimodal_medical.src.datasets.CBIS_DDSM import (
    CBIS_DDSM_dataset,
    CBIS_DDSM_dataset_tfds,
    CBIS_DDSM_whole_mamm_dataset,
    CBIS_DDSM_whole_mamm_breast_density_dataset,
    CBIS_DDSM_whole_mamm_abnormality_dataset,
    CBIS_DDSM_tabular_dataset,
)


def get_datasets(dataset, transform_dict, data_dir=None):
    if dataset == "CBIS-DDSM":
        train_dataset = CBIS_DDSM_dataset(
            data_root="/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/CBIS-DDSM/original",
            # abnormality=['mass'],
            # birads=[3, 4],
            split="train",
            class_type="2classes",
            transform=transform_dict["train"],
        )
        test_dataset = CBIS_DDSM_dataset(
            data_root="/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/CBIS-DDSM/original",
            # abnormality=['mass'],
            # birads=[3, 4],
            split="test",
            class_type="2classes",
            transform=transform_dict["test"],
        )

    elif dataset == "CBIS-DDSM-tfds":
        train_dataset = CBIS_DDSM_dataset_tfds(
            data_root="/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/CBIS-DDSM/original",
            incl_bg=True,
            official_split=True,
            split="train",
            class_type="4classes",
            transform=transform_dict["train"],
        )
        val_dataset = CBIS_DDSM_dataset_tfds(
            data_root="/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/CBIS-DDSM/original",
            incl_bg=True,
            official_split=True,
            split="val",
            class_type="4classes",
            transform=transform_dict["val"],
        )
        test_dataset = CBIS_DDSM_dataset_tfds(
            data_root="/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/CBIS-DDSM/original",
            incl_bg=True,
            official_split=True,
            split="test",
            class_type="4classes",
            transform=transform_dict["test"],
        )

        return train_dataset, val_dataset, test_dataset
    
    elif dataset == "CBIS-DDSM-mass-only-tfds":
        train_dataset = CBIS_DDSM_dataset_tfds(
            data_root="/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/CBIS-DDSM/original",
            incl_bg=False,
            abnormality=['mass'],
            official_split=True,
            split="train",
            class_type="2classes",
            transform=transform_dict["train"],
        )
        val_dataset = CBIS_DDSM_dataset_tfds(
            data_root="/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/CBIS-DDSM/original",
            incl_bg=False,
            abnormality=['mass'],
            official_split=True,
            split="val",
            class_type="2classes",
            transform=transform_dict["val"],
        )
        test_dataset = CBIS_DDSM_dataset_tfds(
            data_root="/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/CBIS-DDSM/original",
            incl_bg=False,
            abnormality=['mass'],
            official_split=True,
            split="test",
            class_type="2classes",
            transform=transform_dict["test"],
        )

        return train_dataset, val_dataset, test_dataset
    
    elif dataset == "CBIS-DDSM-calc-only-tfds":
        train_dataset = CBIS_DDSM_dataset_tfds(
            data_root="/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/CBIS-DDSM/original",
            incl_bg=False,
            abnormality=['calcification'],
            official_split=True,
            split="train",
            class_type="2classes",
            transform=transform_dict["train"],
        )
        val_dataset = CBIS_DDSM_dataset_tfds(
            data_root="/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/CBIS-DDSM/original",
            incl_bg=False,
            abnormality=['calcification'],
            official_split=True,
            split="val",
            class_type="2classes",
            transform=transform_dict["val"],
        )
        test_dataset = CBIS_DDSM_dataset_tfds(
            data_root="/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/CBIS-DDSM/original",
            incl_bg=False,
            abnormality=['calcification'],
            official_split=True,
            split="test",
            class_type="2classes",
            transform=transform_dict["test"],
        )

        return train_dataset, val_dataset, test_dataset
    

    elif dataset == "CBIS-DDSM-tfds-2classes":
        train_dataset = CBIS_DDSM_dataset_tfds(
            data_root="/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/CBIS-DDSM/original",
            incl_bg=False,
            official_split=True,
            split="train",
            class_type="2classes",
            transform=transform_dict["train"],
        )
        val_dataset = CBIS_DDSM_dataset_tfds(
            data_root="/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/CBIS-DDSM/original",
            incl_bg=False,
            official_split=True,
            split="val",
            class_type="2classes",
            transform=transform_dict["val"],
        )
        test_dataset = CBIS_DDSM_dataset_tfds(
            data_root="/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/CBIS-DDSM/original",
            incl_bg=False,
            official_split=True,
            split="test",
            class_type="2classes",
            transform=transform_dict["test"],
        )

        return train_dataset, val_dataset, test_dataset
    elif dataset == "CBIS-DDSM-whole-mamm":
        train_dataset = CBIS_DDSM_whole_mamm_dataset(
            data_root="/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/CBIS-DDSM/original",
            split="train",
            class_type="2classes",
            data_dir=data_dir,
            transform=transform_dict["train"],
        )
        test_dataset = CBIS_DDSM_whole_mamm_dataset(
            data_root="/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/CBIS-DDSM/original",
            split="test",
            class_type="2classes",
            data_dir=data_dir,
            transform=transform_dict["test"],
        )

    elif dataset == "CBIS-DDSM-whole-mamm-breast-density":
        train_dataset = CBIS_DDSM_whole_mamm_breast_density_dataset(
            data_root="/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/CBIS-DDSM/original",
            split="train",
            data_dir=data_dir,
            transform=transform_dict["train"],
        )
        test_dataset = CBIS_DDSM_whole_mamm_breast_density_dataset(
            data_root="/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/CBIS-DDSM/original",
            split="test",
            data_dir=data_dir,
            transform=transform_dict["test"],
        )

    elif dataset == "CBIS-DDSM-whole-mamm-abnormality":
        train_dataset = CBIS_DDSM_whole_mamm_abnormality_dataset(
            data_root="/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/CBIS-DDSM/original",
            split="train",
            data_dir=data_dir,
            transform=transform_dict["train"],
        )
        test_dataset = CBIS_DDSM_whole_mamm_abnormality_dataset(
            data_root="/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/CBIS-DDSM/original",
            split="test",
            data_dir=data_dir,
            transform=transform_dict["test"],
        )

    elif dataset == "CBIS-DDSM-tabular":
        train_dataset = CBIS_DDSM_tabular_dataset(
            data_root="/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/CBIS-DDSM/original",  # birads=[4],
            split="train",
            class_type="2classes",
            methodist_feats_only=False,
        )
        test_dataset = CBIS_DDSM_tabular_dataset(
            data_root="/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/CBIS-DDSM/original",  # birads=[4],
            split="test",
            class_type="2classes",
            methodist_feats_only=False,
        )
    
    elif "CBIS-DDSM-tfds-with-tabular" in dataset:
        if dataset == "CBIS-DDSM-tfds-with-tabular-2classes":
            _birads = [0, 1, 2, 3, 4, 5]
        elif dataset == "CBIS-DDSM-tfds-with-tabular-2classes-birad3":
            _birads = [3]
        elif dataset == "CBIS-DDSM-tfds-with-tabular-2classes-birad4":
            _birads = [4]
        elif dataset == "CBIS-DDSM-tfds-with-tabular-2classes-birad24":
            _birads = [2, 4]
        elif dataset == "CBIS-DDSM-tfds-with-tabular-2classes-birad01245":
            _birads = [0, 1, 2, 4, 5]

        train_dataset = CBIS_DDSM_dataset_tfds(
            data_root="/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/CBIS-DDSM/original",
            incl_bg=False,
            official_split=True,
            split="train",
            birads=_birads,
            class_type="2classes",
            incl_tabular=True,
            transform=transform_dict["train"],
        )
        val_dataset = CBIS_DDSM_dataset_tfds(
            data_root="/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/CBIS-DDSM/original",
            incl_bg=False,
            official_split=True,
            split="val",
            birads=_birads,
            class_type="2classes",
            incl_tabular=True,
            transform=transform_dict["val"],
        )
        test_dataset = CBIS_DDSM_dataset_tfds(
            data_root="/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/CBIS-DDSM/original",
            incl_bg=False,
            official_split=True,
            split="test",
            birads=_birads,
            class_type="2classes",
            incl_tabular=True,
            transform=transform_dict["test"],
        )

        return train_dataset, val_dataset, test_dataset
    
    elif dataset == "CBIS-DDSM-tfds-with-tabular-methodist-2classes":
        train_dataset = CBIS_DDSM_dataset_tfds(
            data_root="/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/CBIS-DDSM/original",
            incl_bg=False,
            official_split=True,
            split="train",
            class_type="2classes",
            incl_tabular=True,
            tabular_methodist_feats_only=True,
            transform=transform_dict["train"],
        )
        val_dataset = CBIS_DDSM_dataset_tfds(
            data_root="/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/CBIS-DDSM/original",
            incl_bg=False,
            official_split=True,
            split="val",
            class_type="2classes",
            incl_tabular=True,
            tabular_methodist_feats_only=True,
            transform=transform_dict["val"],
        )
        test_dataset = CBIS_DDSM_dataset_tfds(
            data_root="/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/CBIS-DDSM/original",
            incl_bg=False,
            official_split=True,
            split="test",
            class_type="2classes",
            incl_tabular=True,
            tabular_methodist_feats_only=True,
            transform=transform_dict["test"],
        )

        return train_dataset, val_dataset, test_dataset
    
    elif dataset == "CBIS-DDSM-tfds-with-tabular-methodist-mass-appearance":
        train_dataset = CBIS_DDSM_dataset_tfds(
            data_root="/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/CBIS-DDSM/original",
            incl_bg=False,
            official_split=True,
            split="train",
            abnormality=["mass"],
            class_type="2classes",
            incl_tabular=True,
            tabular_methodist_feats_only=True,
            label_type="mass_appearance",
            transform=transform_dict["train"],
        )
        val_dataset = CBIS_DDSM_dataset_tfds(
            data_root="/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/CBIS-DDSM/original",
            incl_bg=False,
            official_split=True,
            split="val",
            abnormality=["mass"],
            class_type="2classes",
            incl_tabular=True,
            tabular_methodist_feats_only=True,
            label_type="mass_appearance",
            transform=transform_dict["val"],
        )
        test_dataset = CBIS_DDSM_dataset_tfds(
            data_root="/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/CBIS-DDSM/original",
            incl_bg=False,
            official_split=True,
            split="test",
            abnormality=["mass"],
            class_type="2classes",
            incl_tabular=True,
            tabular_methodist_feats_only=True,
            label_type="mass_appearance",
            transform=transform_dict["test"],
        )

        return train_dataset, val_dataset, test_dataset
    
    elif dataset == "CBIS-DDSM-tfds-with-tabular-methodist-calc-morph":
        train_dataset = CBIS_DDSM_dataset_tfds(
            data_root="/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/CBIS-DDSM/original",
            incl_bg=False,
            official_split=True,
            split="train",
            abnormality=["calcification"],
            class_type="2classes",
            incl_tabular=True,
            tabular_methodist_feats_only=True,
            label_type="calc_morph",
            transform=transform_dict["train"],
        )
        val_dataset = CBIS_DDSM_dataset_tfds(
            data_root="/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/CBIS-DDSM/original",
            incl_bg=False,
            official_split=True,
            split="val",
            abnormality=["calcification"],
            class_type="2classes",
            incl_tabular=True,
            tabular_methodist_feats_only=True,
            label_type="calc_morph",
            transform=transform_dict["val"],
        )
        test_dataset = CBIS_DDSM_dataset_tfds(
            data_root="/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/CBIS-DDSM/original",
            incl_bg=False,
            official_split=True,
            split="test",
            abnormality=["calcification"],
            class_type="2classes",
            incl_tabular=True,
            tabular_methodist_feats_only=True,
            label_type="calc_morph",
            transform=transform_dict["test"],
        )

        return train_dataset, val_dataset, test_dataset

    elif dataset == "CBIS-DDSM-tfds-with-tabular-methodist-calc-dist":
        train_dataset = CBIS_DDSM_dataset_tfds(
            data_root="/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/CBIS-DDSM/original",
            incl_bg=False,
            official_split=True,
            split="train",
            abnormality=["calcification"],
            class_type="2classes",
            incl_tabular=True,
            tabular_methodist_feats_only=True,
            label_type="calc_dist",
            transform=transform_dict["train"],
        )
        val_dataset = CBIS_DDSM_dataset_tfds(
            data_root="/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/CBIS-DDSM/original",
            incl_bg=False,
            official_split=True,
            split="val",
            abnormality=["calcification"],
            class_type="2classes",
            incl_tabular=True,
            tabular_methodist_feats_only=True,
            label_type="calc_dist",
            transform=transform_dict["val"],
        )
        test_dataset = CBIS_DDSM_dataset_tfds(
            data_root="/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/CBIS-DDSM/original",
            incl_bg=False,
            official_split=True,
            split="test",
            abnormality=["calcification"],
            class_type="2classes",
            incl_tabular=True,
            tabular_methodist_feats_only=True,
            label_type="calc_dist",
            transform=transform_dict["test"],
        )

        return train_dataset, val_dataset, test_dataset
    
    elif dataset == "CBIS-DDSM-tfds-with-tabular-mass-shape":
        train_dataset = CBIS_DDSM_dataset_tfds(
            data_root="/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/CBIS-DDSM/original",
            incl_bg=False,
            official_split=True,
            split="train",
            abnormality=["mass"],
            class_type="2classes",
            incl_tabular=True,
            tabular_methodist_feats_only=False,
            label_type="mass_shape",
            transform=transform_dict["train"],
        )
        val_dataset = CBIS_DDSM_dataset_tfds(
            data_root="/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/CBIS-DDSM/original",
            incl_bg=False,
            official_split=True,
            split="val",
            abnormality=["mass"],
            class_type="2classes",
            incl_tabular=True,
            tabular_methodist_feats_only=False,
            label_type="mass_shape",
            transform=transform_dict["val"],
        )
        test_dataset = CBIS_DDSM_dataset_tfds(
            data_root="/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/CBIS-DDSM/original",
            incl_bg=False,
            official_split=True,
            split="test",
            abnormality=["mass"],
            class_type="2classes",
            incl_tabular=True,
            tabular_methodist_feats_only=False,
            label_type="mass_shape",
            transform=transform_dict["test"],
        )

        return train_dataset, val_dataset, test_dataset

    elif dataset == "CBIS-DDSM-tfds-with-tabular-mass-margin":
        train_dataset = CBIS_DDSM_dataset_tfds(
            data_root="/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/CBIS-DDSM/original",
            incl_bg=False,
            official_split=True,
            split="train",
            abnormality=["mass"],
            class_type="2classes",
            incl_tabular=True,
            tabular_methodist_feats_only=False,
            label_type="mass_margin",
            transform=transform_dict["train"],
        )
        val_dataset = CBIS_DDSM_dataset_tfds(
            data_root="/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/CBIS-DDSM/original",
            incl_bg=False,
            official_split=True,
            split="val",
            abnormality=["mass"],
            class_type="2classes",
            incl_tabular=True,
            tabular_methodist_feats_only=False,
            label_type="mass_margin",
            transform=transform_dict["val"],
        )
        test_dataset = CBIS_DDSM_dataset_tfds(
            data_root="/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/CBIS-DDSM/original",
            incl_bg=False,
            official_split=True,
            split="test",
            abnormality=["mass"],
            class_type="2classes",
            incl_tabular=True,
            tabular_methodist_feats_only=False,
            label_type="mass_margin",
            transform=transform_dict["test"],
        )

        return train_dataset, val_dataset, test_dataset
    
    elif dataset == "CBIS-DDSM-tfds-with-tabular-calc-morph":
        train_dataset = CBIS_DDSM_dataset_tfds(
            data_root="/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/CBIS-DDSM/original",
            incl_bg=False,
            official_split=True,
            split="train",
            abnormality=["calcification"],
            class_type="2classes",
            incl_tabular=True,
            tabular_methodist_feats_only=False,
            label_type="calc_morph",
            transform=transform_dict["train"],
        )
        val_dataset = CBIS_DDSM_dataset_tfds(
            data_root="/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/CBIS-DDSM/original",
            incl_bg=False,
            official_split=True,
            split="val",
            abnormality=["calcification"],
            class_type="2classes",
            incl_tabular=True,
            tabular_methodist_feats_only=False,
            label_type="calc_morph",
            transform=transform_dict["val"],
        )
        test_dataset = CBIS_DDSM_dataset_tfds(
            data_root="/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/CBIS-DDSM/original",
            incl_bg=False,
            official_split=True,
            split="test",
            abnormality=["calcification"],
            class_type="2classes",
            incl_tabular=True,
            tabular_methodist_feats_only=False,
            label_type="calc_morph",
            transform=transform_dict["test"],
        )

        return train_dataset, val_dataset, test_dataset
    
    elif dataset == "CBIS-DDSM-tfds-with-tabular-calc-dist":
        train_dataset = CBIS_DDSM_dataset_tfds(
            data_root="/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/CBIS-DDSM/original",
            incl_bg=False,
            official_split=True,
            split="train",
            abnormality=["calcification"],
            class_type="2classes",
            incl_tabular=True,
            tabular_methodist_feats_only=False,
            label_type="calc_dist",
            transform=transform_dict["train"],
        )
        val_dataset = CBIS_DDSM_dataset_tfds(
            data_root="/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/CBIS-DDSM/original",
            incl_bg=False,
            official_split=True,
            split="val",
            abnormality=["calcification"],
            class_type="2classes",
            incl_tabular=True,
            tabular_methodist_feats_only=False,
            label_type="calc_dist",
            transform=transform_dict["val"],
        )
        test_dataset = CBIS_DDSM_dataset_tfds(
            data_root="/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/CBIS-DDSM/original",
            incl_bg=False,
            official_split=True,
            split="test",
            abnormality=["calcification"],
            class_type="2classes",
            incl_tabular=True,
            tabular_methodist_feats_only=False,
            label_type="calc_dist",
            transform=transform_dict["test"],
        )

        return train_dataset, val_dataset, test_dataset

   
    else:
        raise ValueError(dataset)

    return train_dataset, test_dataset


def get_combined_datasets(
    train_datasets_list, val_datasets_list, test_datasets_list, transform_dict,
    train_datasets_dir_list, val_datasets_dir_list, test_datasets_dir_list,
    train_partitions_list=None, val_partitions_list=None, test_partitions_list=None
):
    all_train_datasets, all_val_datasets, all_test_datasets = [], [], []

    datasets_with_train_val_test = \
                                [
                                    "CBIS-DDSM-tfds",
                                    'CBIS-DDSM-tfds-2classes',                                    
                                    "CBIS-DDSM-mass-only-tfds",                                    
                                    "CBIS-DDSM-calc-only-tfds",
                                    
                                    "CBIS-DDSM-tfds-with-tabular-mass-margin",
                                    "CBIS-DDSM-tfds-with-tabular-calc-morph",
                                    "CBIS-DDSM-tfds-with-tabular-2classes",
                                    "CBIS-DDSM-tfds-with-tabular-2classes-birad3",
                                    "CBIS-DDSM-tfds-with-tabular-2classes-birad4",
                                    "CBIS-DDSM-tfds-with-tabular-2classes-birad24",
                                    "CBIS-DDSM-tfds-with-tabular-2classes-birad01245",

                                    "CBIS-DDSM-tfds-with-tabular-methodist-2classes",
                                    "CBIS-DDSM-tfds-with-tabular-methodist-mass-appearance",
                                    "CBIS-DDSM-tfds-with-tabular-methodist-calc-morph",
                                    "CBIS-DDSM-tfds-with-tabular-methodist-calc-dist",

                                    "CBIS-DDSM-tfds-with-tabular-mass-shape",
                                    "CBIS-DDSM-tfds-with-tabular-mass-margin",
                                    "CBIS-DDSM-tfds-with-tabular-calc-morph",
                                    "CBIS-DDSM-tfds-with-tabular-calc-dist",                                                                        
                                ]


    partition_mapping = {
        'w_tvt': { # for datasets with train/val/test
            'train': 0,
            'val': 1,
            'test': 2
        },
        'wo_tvt': { # for datasets without train/val/test
            'train': 0,
            'test': 1
        }
    }

    if train_partitions_list is None:
        train_partitions_list = ['train'] * len(train_datasets_list)
    if val_partitions_list is None:
        val_partitions_list = ['val'] * len(val_datasets_list)
    if test_partitions_list is None:
        test_partitions_list = ['test'] * len(test_datasets_list)

    for dataset, dataset_dir, train_partition in zip(train_datasets_list, train_datasets_dir_list, train_partitions_list):
        if dataset in datasets_with_train_val_test:
            dataset_splits = get_datasets(dataset, transform_dict)
            part_idx = partition_mapping['w_tvt'][train_partition]
            train_dataset = dataset_splits[part_idx]
        else:
            dataset_splits = get_datasets(dataset, transform_dict, dataset_dir)
            part_idx = partition_mapping['wo_tvt'][train_partition]
            train_dataset = dataset_splits[part_idx]

        all_train_datasets.append(train_dataset)

    for dataset, dataset_dir, val_partition in zip(val_datasets_list, val_datasets_dir_list, val_partitions_list):
        if dataset in datasets_with_train_val_test:
            dataset_splits = get_datasets(dataset, transform_dict)
            part_idx = partition_mapping['w_tvt'][val_partition]
            val_dataset = dataset_splits[part_idx]
        else:
            dataset_splits = get_datasets(dataset, transform_dict, dataset_dir)
            part_idx = partition_mapping['wo_tvt'][val_partition]
            val_dataset = dataset_splits[part_idx]

        all_val_datasets.append(val_dataset)

    for dataset, dataset_dir, test_partition in zip(test_datasets_list, test_datasets_dir_list, test_partitions_list):
        if dataset in datasets_with_train_val_test:
            dataset_splits = get_datasets(dataset, transform_dict)
            part_idx = partition_mapping['w_tvt'][test_partition]
            test_dataset = dataset_splits[part_idx]
        else:
            dataset_splits = get_datasets(dataset, transform_dict, dataset_dir)
            part_idx = partition_mapping['wo_tvt'][test_partition]
            test_dataset = dataset_splits[part_idx]

        all_test_datasets.append(test_dataset)


    return all_train_datasets, all_val_datasets, all_test_datasets