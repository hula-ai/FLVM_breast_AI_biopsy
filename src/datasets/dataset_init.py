from MultiMEDal_multimodal_medical.src.datasets.CBIS_DDSM import CBIS_DDSM_dataset_tfds
from MultiMEDal_multimodal_medical.src.datasets.EMBED import EMBED_dataset_tfds


def get_datasets(dataset, transform_dict, data_dir=None):

    if dataset == "CBIS-DDSM-tfds-with-tabular-2classes":
        train_dataset = CBIS_DDSM_dataset_tfds(
            data_root="/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/CBIS-DDSM/original",
            incl_bg=False,
            official_split=True,
            split="train",
            class_type="2classes",
            incl_tabular=True,
            transform=transform_dict["train"],
        )
        val_dataset = CBIS_DDSM_dataset_tfds(
            data_root="/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/CBIS-DDSM/original",
            incl_bg=False,
            official_split=True,
            split="val",
            class_type="2classes",
            incl_tabular=True,
            transform=transform_dict["val"],
        )
        test_dataset = CBIS_DDSM_dataset_tfds(
            data_root="/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/CBIS-DDSM/original",
            incl_bg=False,
            official_split=True,
            split="test",
            class_type="2classes",
            incl_tabular=True,
            transform=transform_dict["test"],
        )

        return train_dataset, val_dataset, test_dataset

    elif dataset == "EMBED-unique-mapping-tfds-with-tabular-2classes":
        train_dataset = EMBED_dataset_tfds(
            data_root="/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/EMBED/original",
            dataset_dir="images_unique_mapping_tfds",
            incl_bg=False,
            split="train",
            class_type="2classes",
            incl_tabular=True,
            transform=transform_dict["train"],
        )
        val_dataset = EMBED_dataset_tfds(
            data_root="/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/EMBED/original",
            dataset_dir="images_unique_mapping_tfds",
            incl_bg=False,
            split="test",
            class_type="2classes",
            incl_tabular=True,
            transform=transform_dict["val"],
        )
        test_dataset = EMBED_dataset_tfds(
            data_root="/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/EMBED/original",
            dataset_dir="images_unique_mapping_tfds",
            incl_bg=False,
            split="test",
            class_type="2classes",
            incl_tabular=True,
            transform=transform_dict["test"],
        )

        return train_dataset, val_dataset, test_dataset

    else:
        raise ValueError(dataset)

    return train_dataset, test_dataset


def get_combined_datasets(
    train_datasets_list,
    val_datasets_list,
    test_datasets_list,
    transform_dict,
    train_datasets_dir_list,
    val_datasets_dir_list,
    test_datasets_dir_list,
):
    all_train_datasets, all_val_datasets, all_test_datasets = [], [], []

    datasets_with_train_val_test = [
        "CBIS-DDSM-tfds-with-tabular-2classes",
        "CBIS-DDSM-tfds-with-tabular-2classes-birad3",
        "EMBED-unique-mapping-tfds-with-tabular-2classes",
    ]

    for dataset, dataset_dir in zip(train_datasets_list, train_datasets_dir_list):
        if dataset in datasets_with_train_val_test:
            train_dataset, _, _ = get_datasets(dataset, transform_dict)
        else:
            train_dataset, _ = get_datasets(dataset, transform_dict, dataset_dir)
        all_train_datasets.append(train_dataset)

    for dataset, dataset_dir in zip(val_datasets_list, val_datasets_dir_list):
        if dataset in datasets_with_train_val_test:
            _, test_dataset, _ = get_datasets(dataset, transform_dict)
        else:
            _, test_dataset = get_datasets(dataset, transform_dict, dataset_dir)
        all_val_datasets.append(test_dataset)

    for dataset, dataset_dir in zip(test_datasets_list, test_datasets_dir_list):
        if dataset in datasets_with_train_val_test:
            _, _, test_dataset = get_datasets(dataset, transform_dict)
        else:
            _, test_dataset = get_datasets(dataset, transform_dict, dataset_dir)
        all_test_datasets.append(test_dataset)

    return all_train_datasets, all_val_datasets, all_test_datasets
