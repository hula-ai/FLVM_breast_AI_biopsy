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


    elif dataset == "EMBED":
        train_dataset = EMBED_dataset(
            data_root="/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/EMBED/original",
            split="train",
            class_type="2classes",
            transform=transform_dict["train"],
        )
        test_dataset = EMBED_dataset(
            data_root="/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/EMBED/original",
            split="test",
            class_type="2classes",
            transform=transform_dict["test"],
        )
    elif dataset == "EMBED-whole-mamm":
        train_dataset = EMBED_whole_mamm_dataset(
            data_root="/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/EMBED/original",
            split="train",
            class_type="2classes",
            data_dir=data_dir,
            transform=transform_dict["train"],
        )
        test_dataset = EMBED_whole_mamm_dataset(
            data_root="/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/EMBED/original",
            split="test",
            class_type="2classes",
            data_dir=data_dir,
            transform=transform_dict["test"],
        )
    elif dataset == "EMBED-whole-mamm-unique-mapping":
        train_dataset = EMBED_whole_mamm_dataset(
            data_root="/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/EMBED/original",
            split="train",
            class_type="2classes",
            data_dir=data_dir,
            unique_mapping=True,
            transform=transform_dict["train"],
        )
        test_dataset = EMBED_whole_mamm_dataset(
            data_root="/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/EMBED/original",
            split="test",
            class_type="2classes",
            data_dir=data_dir,
            unique_mapping=True,
            transform=transform_dict["test"],
        )
    elif dataset == "EMBED-tfds":
        train_dataset = EMBED_dataset_tfds(
            data_root="/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/EMBED/original",
            incl_bg=True,
            split="train",
            class_type="4classes",
            transform=transform_dict["train"],
        )
        val_dataset = EMBED_dataset_tfds(
            data_root="/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/EMBED/original",
            incl_bg=True,
            split="test",
            class_type="4classes",
            transform=transform_dict["val"],
        )
        test_dataset = EMBED_dataset_tfds(
            data_root="/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/EMBED/original",
            incl_bg=True,
            split="test",
            class_type="4classes",
            transform=transform_dict["test"],
        )

        return train_dataset, val_dataset, test_dataset

    elif dataset == "EMBED-unique-mapping-tfds":
        train_dataset = EMBED_dataset_tfds(
            data_root="/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/EMBED/original",
            dataset_dir="images_unique_mapping_tfds",
            incl_bg=True,
            split="train",
            class_type="4classes",
            transform=transform_dict["train"],
        )
        val_dataset = EMBED_dataset_tfds(
            data_root="/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/EMBED/original",
            dataset_dir="images_unique_mapping_tfds",
            incl_bg=True,
            split="test",
            class_type="4classes",
            transform=transform_dict["val"],
        )
        test_dataset = EMBED_dataset_tfds(
            data_root="/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/EMBED/original",
            dataset_dir="images_unique_mapping_tfds",
            incl_bg=True,
            split="test",
            class_type="4classes",
            transform=transform_dict["test"],
        )

        return train_dataset, val_dataset, test_dataset
    
    elif dataset == "EMBED-unique-mapping-mass-only-tfds":
        train_dataset = EMBED_dataset_tfds(
            data_root="/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/EMBED/original",
            dataset_dir="images_unique_mapping_tfds",
            incl_bg=False,
            abnormality=['mass'],
            split="train",
            class_type="2classes",
            transform=transform_dict["train"],
        )
        val_dataset = EMBED_dataset_tfds(
            data_root="/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/EMBED/original",
            dataset_dir="images_unique_mapping_tfds",
            incl_bg=False,
            abnormality=['mass'],
            split="test",
            class_type="2classes",
            transform=transform_dict["val"],
        )
        test_dataset = EMBED_dataset_tfds(
            data_root="/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/EMBED/original",
            dataset_dir="images_unique_mapping_tfds",
            incl_bg=False,
            abnormality=['mass'],
            split="test",
            class_type="2classes",
            transform=transform_dict["test"],
        )

        return train_dataset, val_dataset, test_dataset
    
    elif dataset == "EMBED-unique-mapping-calc-only-tfds":
        train_dataset = EMBED_dataset_tfds(
            data_root="/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/EMBED/original",
            dataset_dir="images_unique_mapping_tfds",
            incl_bg=False,
            abnormality=['calcification'],
            split="train",
            class_type="2classes",
            transform=transform_dict["train"],
        )
        val_dataset = EMBED_dataset_tfds(
            data_root="/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/EMBED/original",
            dataset_dir="images_unique_mapping_tfds",
            incl_bg=False,
            abnormality=['calcification'],
            split="test",
            class_type="2classes",
            transform=transform_dict["val"],
        )
        test_dataset = EMBED_dataset_tfds(
            data_root="/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/EMBED/original",
            dataset_dir="images_unique_mapping_tfds",
            incl_bg=False,
            abnormality=['calcification'],
            split="test",
            class_type="2classes",
            transform=transform_dict["test"],
        )

        return train_dataset, val_dataset, test_dataset

    elif dataset == "EMBED-tfds-2classes":
        train_dataset = EMBED_dataset_tfds(
            data_root="/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/EMBED/original",
            incl_bg=False,
            split="train",
            class_type="2classes",
            transform=transform_dict["train"],
        )
        val_dataset = EMBED_dataset_tfds(
            data_root="/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/EMBED/original",
            incl_bg=False,
            split="test",
            class_type="2classes",
            transform=transform_dict["val"],
        )
        test_dataset = EMBED_dataset_tfds(
            data_root="/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/EMBED/original",
            incl_bg=False,
            split="test",
            class_type="2classes",
            transform=transform_dict["test"],
        )

        return train_dataset, val_dataset, test_dataset
    
    elif dataset == "EMBED-unique-mapping-tfds-2classes":
        train_dataset = EMBED_dataset_tfds(
            data_root="/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/EMBED/original",
            dataset_dir="images_unique_mapping_tfds",
            incl_bg=False,
            split="train",
            class_type="2classes",
            transform=transform_dict["train"],
        )
        val_dataset = EMBED_dataset_tfds(
            data_root="/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/EMBED/original",
            dataset_dir="images_unique_mapping_tfds",
            incl_bg=False,
            split="test",
            class_type="2classes",
            transform=transform_dict["val"],
        )
        test_dataset = EMBED_dataset_tfds(
            data_root="/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/EMBED/original",
            dataset_dir="images_unique_mapping_tfds",
            incl_bg=False,
            split="test",
            class_type="2classes",
            transform=transform_dict["test"],
        )

        return train_dataset, val_dataset, test_dataset

    elif dataset == "CDD-CESM-whole-mamm":
        train_dataset = CDD_CESM_whole_mamm_dataset(
            data_root='/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/CDD-CESM/original',
            split='train',
            class_type="2classes",
            data_dir=data_dir,
            transform=transform_dict['train']
        )
        test_dataset = CDD_CESM_whole_mamm_dataset(
            data_root='/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/CDD-CESM/original',
            split='test',
            class_type="2classes",
            data_dir=data_dir,
            transform=transform_dict['test']
        )
    elif dataset == "CDD-CESM-DM-whole-mamm":
        train_dataset = CDD_CESM_whole_mamm_dataset(
            data_root='/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/CDD-CESM/original',
            split='train',
            class_type="2classes",
            img_type=["DM"],
            data_dir=data_dir,
            transform=transform_dict['train']
        )
        test_dataset = CDD_CESM_whole_mamm_dataset(
            data_root='/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/CDD-CESM/original',
            split='test',
            class_type="2classes",
            img_type=["DM"],
            data_dir=data_dir,
            transform=transform_dict['test']
        )
    elif dataset == "CDD-CESM-CE-whole-mamm":
        train_dataset = CDD_CESM_whole_mamm_dataset(
            data_root='/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/CDD-CESM/original',
            split='train',
            class_type="2classes",
            img_type=["CESM"],
            data_dir=data_dir,
            transform=transform_dict['train']
        )
        test_dataset = CDD_CESM_whole_mamm_dataset(
            data_root='/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/CDD-CESM/original',
            split='test',
            class_type="2classes",
            img_type=["CESM"],
            data_dir=data_dir,
            transform=transform_dict['test']
        )
    elif dataset == "CDD-CESM-whole-mamm-train-only":
        train_dataset = CDD_CESM_whole_mamm_dataset(
            data_root='/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/CDD-CESM/original',
            class_type="2classes",
            data_dir=data_dir,
            transform=transform_dict['train']
        )
        test_dataset = None
    elif dataset == "CDD-CESM-whole-mamm-test-only":
        train_dataset = None
        test_dataset = CDD_CESM_whole_mamm_dataset(
            data_root='/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/CDD-CESM/original',
            class_type="2classes",
            data_dir=data_dir,
            transform=transform_dict['test']
        )        
    elif dataset == "CDD-CESM-DM-whole-mamm-test-only":
        train_dataset = None
        test_dataset = CDD_CESM_whole_mamm_dataset(
            data_root='/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/CDD-CESM/original',
            class_type="2classes",
            img_type=["DM"],
            data_dir=data_dir,
            transform=transform_dict['test']
        )        
    elif dataset == 'CSAWCC-whole-mamm':
        train_dataset = CSAWCC_whole_mamm_dataset(
            data_root='/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/CSAWCC/original',
            class_type="2classes",
            split='train',
            data_dir=data_dir,
            transform=transform_dict['train']
        )
        test_dataset = CSAWCC_whole_mamm_dataset(
            data_root='/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/CSAWCC/original',
            class_type="2classes",
            split='test',
            data_dir=data_dir,
            transform=transform_dict['test']
        )
    elif dataset == 'CSAWCC-whole-mamm-train-only':
        train_dataset = CSAWCC_whole_mamm_dataset(
            data_root='/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/CSAWCC/original',
            class_type="2classes",
            data_dir=data_dir,
            transform=transform_dict['train']
        )
        test_dataset = None
    elif dataset == 'CSAWCC-whole-mamm-test-only':
        train_dataset = None
        test_dataset = CSAWCC_whole_mamm_dataset(
            data_root='/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/CSAWCC/original',
            class_type="2classes",
            data_dir=data_dir,
            transform=transform_dict['test']
        )
    elif dataset == "BMCD-whole-mamm":
        train_dataset = BMCD_whole_mamm_dataset(
            data_root="/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/BMCD/original",
            split="train",
            class_type="2classes",
            data_dir=data_dir,
            transform=transform_dict["train"],
        )
        test_dataset = BMCD_whole_mamm_dataset(
            data_root="/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/BMCD/original",
            split="test",
            class_type="2classes",
            data_dir=data_dir,
            transform=transform_dict["test"],
        )
    elif dataset == 'INbreast-whole-mamm-train-only':
        train_dataset = INbreast_whole_mamm_dataset(
            data_root='/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/INbreast',
            class_type="2classes",
            data_dir=data_dir,
            transform=transform_dict['train']
        )
        test_dataset = None
    elif dataset == 'INbreast-whole-mamm-test-only':
        train_dataset = None
        test_dataset = INbreast_whole_mamm_dataset(
            data_root='/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/INbreast',
            class_type="2classes",
            data_dir=data_dir,
            transform=transform_dict['test']
        )
    elif dataset == "INbreast-tfds":
        train_dataset = INbreast_dataset_tfds(
            data_root="/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/INbreast/INbreast_images_pngs/",
            incl_bg=True,
            class_type="4classes",
            transform=transform_dict["train"],
        )
        val_dataset = None
        test_dataset = None

        return train_dataset, val_dataset, test_dataset
    elif dataset == "INbreast-mass-only-tfds":
        train_dataset = INbreast_dataset_tfds(
            data_root="/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/INbreast/INbreast_images_pngs/",
            incl_bg=False,
            abnormality=['mass'],
            class_type="2classes",
            transform=transform_dict["train"],
        )
        val_dataset = INbreast_dataset_tfds(
            data_root="/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/INbreast/INbreast_images_pngs/",
            incl_bg=False,
            abnormality=['mass'],
            class_type="2classes",
            transform=transform_dict["val"],
        )
        test_dataset = INbreast_dataset_tfds(
            data_root="/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/INbreast/INbreast_images_pngs/",
            incl_bg=False,
            abnormality=['mass'],
            class_type="2classes",
            transform=transform_dict["test"],
        )

        return train_dataset, val_dataset, test_dataset
    
    elif dataset == "INbreast-calc-only-tfds":
        train_dataset = INbreast_dataset_tfds(
            data_root="/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/INbreast/INbreast_images_pngs/",
            incl_bg=False,
            abnormality=['calcification'],
            class_type="2classes",
            transform=transform_dict["train"],
        )
        val_dataset = INbreast_dataset_tfds(
            data_root="/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/INbreast/INbreast_images_pngs/",
            incl_bg=False,
            abnormality=['calcification'],
            class_type="2classes",
            transform=transform_dict["val"],
        )
        test_dataset = INbreast_dataset_tfds(
            data_root="/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/INbreast/INbreast_images_pngs/",
            incl_bg=False,
            abnormality=['calcification'],
            class_type="2classes",
            transform=transform_dict["test"],
        )

        return train_dataset, val_dataset, test_dataset
    
    elif dataset == "BCDR-digital":
        train_dataset = BCDR_whole_mamm_dataset(
            data_root='/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/BCDR',
            data_type='digital',
            class_type="2classes",
            data_dir=data_dir,
            transform=transform_dict['train']
        )
        test_dataset = None
    elif dataset == "BCDR-digital":
        train_dataset = BCDR_whole_mamm_dataset(
            data_root='/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/BCDR',
            data_type='digital',
            class_type="2classes",
            data_dir=data_dir,
            transform=transform_dict['train']
        )
        test_dataset = None
    
    
    elif dataset == "BCDR-film":
        train_dataset = BCDR_whole_mamm_dataset(
            data_root='/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/BCDR',
            data_type='film',
            class_type="2classes",
            data_dir=data_dir,
            transform=transform_dict['train']
        )
        test_dataset = None
    elif dataset == "BCDR-digital-tfds":
        train_dataset = BCDR_dataset_tfds(
            data_root='/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/BCDR',
            data_type='digital',
            incl_bg=True,
            class_type="4classes",
            transform=transform_dict["train"],
        )
        val_dataset = None
        test_dataset = None

        return train_dataset, val_dataset, test_dataset

    elif dataset == "BCDR-digital-mass-only-tfds":
        train_dataset = BCDR_dataset_tfds(
            data_root='/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/BCDR',
            data_type='digital',
            incl_bg=False,
            abnormality=['mass'],
            class_type="2classes",
            transform=transform_dict["train"],
        )
        val_dataset = BCDR_dataset_tfds(
            data_root='/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/BCDR',
            data_type='digital',
            incl_bg=False,
            abnormality=['mass'],
            class_type="2classes",
            transform=transform_dict["val"],
        )
        test_dataset = BCDR_dataset_tfds(
            data_root='/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/BCDR',
            data_type='digital',
            incl_bg=False,
            abnormality=['mass'],
            class_type="2classes",
            transform=transform_dict["test"],
        )

        return train_dataset, val_dataset, test_dataset
    
    elif dataset == "BCDR-digital-calc-only-tfds":
        train_dataset = BCDR_dataset_tfds(
            data_root='/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/BCDR',
            data_type='digital',
            incl_bg=False,
            abnormality=['calcification'],
            class_type="2classes",
            transform=transform_dict["train"],
        )
        val_dataset = None
        test_dataset = None

        return train_dataset, val_dataset, test_dataset
    

    elif dataset == "BCDR-film-tfds":
        train_dataset = BCDR_dataset_tfds(
            data_root='/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/BCDR',
            data_type='film',
            incl_bg=True,
            class_type="4classes",
            transform=transform_dict["train"],
        )
        val_dataset = None
        test_dataset = None

        return train_dataset, val_dataset, test_dataset

    elif dataset == "BCDR-film-mass-only-tfds":
        train_dataset = BCDR_dataset_tfds(
            data_root='/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/BCDR',
            data_type='film',
            incl_bg=False,
            abnormality=['mass'],
            class_type="2classes",
            transform=transform_dict["train"],
        )
        val_dataset = None
        test_dataset = None

        return train_dataset, val_dataset, test_dataset
    
    elif dataset == "BCDR-film-calc-only-tfds":
        train_dataset = BCDR_dataset_tfds(
            data_root='/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/BCDR',
            data_type='film',
            incl_bg=False,
            abnormality=['calcification'],
            class_type="2classes",
            transform=transform_dict["train"],
        )
        val_dataset = None
        test_dataset = None

        return train_dataset, val_dataset, test_dataset
    
    elif dataset == 'ADMANI_RSNA_site_1':
        train_dataset = ADMANI_RSNA_whole_mamm_dataset(
            data_root='/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/ADMANI_rsna',
            split="train_split",
            site_list=[1],
            class_type="2classes",
            data_dir=data_dir,
            transform=transform_dict["train"],
        )
        test_dataset = ADMANI_RSNA_whole_mamm_dataset(
            data_root='/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/ADMANI_rsna',
            split="test_split",
            site_list=[1],
            class_type="2classes",
            data_dir=data_dir,
            transform=transform_dict["test"],
        )

    elif dataset == 'ADMANI_RSNA_site_2':
        train_dataset = ADMANI_RSNA_whole_mamm_dataset(
            data_root='/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/ADMANI_rsna',
            split="train_split",
            site_list=[2],
            class_type="2classes",
            data_dir=data_dir,
            transform=transform_dict["train"],
        )
        test_dataset = ADMANI_RSNA_whole_mamm_dataset(
            data_root='/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/ADMANI_rsna',
            split="test_split",
            site_list=[2],
            class_type="2classes",
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
    elif dataset == "Methodist-tabular":
        train_dataset = Methodist_tabular_dataset(
            data_root="/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/Methodist",
            split="train",
            class_type="2classes",
            methodist_feats_only=False,
            birads=[4],
        )
        test_dataset = Methodist_tabular_dataset(
            data_root="/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/Methodist",
            split="test",
            class_type="2classes",
            methodist_feats_only=False,
            birads=[4],
        )
    elif dataset == "EMBED-tabular":
        train_dataset = EMBED_tabular_dataset(
            data_root="/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/EMBED/original",
            class_type="2classes",
            methodist_feats_only=False
        )

        test_dataset = EMBED_tabular_dataset(
            data_root="/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/EMBED/original",
            class_type="2classes",
            methodist_feats_only=False
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

    elif dataset == "EMBED-unique-mapping-tfds-with-tabular-demography-only-2classes":
        train_dataset = EMBED_dataset_tfds(
            data_root="/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/EMBED/original",
            dataset_dir="images_unique_mapping_tfds",
            incl_bg=False,
            split="train",
            class_type="2classes",
            incl_tabular=True,
            incl_findings=False,
            incl_breast_dense=False,
            incl_breast_side=False,
            incl_findings_feats=False,
            incl_demography=True,
            transform=transform_dict["train"],
        )
        val_dataset = EMBED_dataset_tfds(
            data_root="/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/EMBED/original",
            dataset_dir="images_unique_mapping_tfds",
            incl_bg=False,
            split="test",
            class_type="2classes",
            incl_tabular=True,
            incl_findings=False,
            incl_breast_dense=False,
            incl_breast_side=False,
            incl_findings_feats=False,
            incl_demography=True,
            transform=transform_dict["val"],
        )
        test_dataset = EMBED_dataset_tfds(
            data_root="/home/hqvo2/Projects/MultiMEDal_multimodal_medical/data/EMBED/original",
            dataset_dir="images_unique_mapping_tfds",
            incl_bg=False,
            split="test",
            class_type="2classes",
            incl_tabular=True,
            incl_findings=False,
            incl_breast_dense=False,
            incl_breast_side=False,
            incl_findings_feats=False,
            incl_demography=True,
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

    datasets_with_train_val_test = ["CBIS-DDSM-tfds", "EMBED-tfds", "EMBED-unique-mapping-tfds", "INbreast-tfds", 
                                    'CBIS-DDSM-tfds-2classes', 'EMBED-tfds-2classes', 'EMBED-unique-mapping-tfds-2classes',
                                    'BCDR-digital-tfds', 'BCDR-film-tfds',
                                    "CBIS-DDSM-mass-only-tfds", "EMBED-mass-only-tfds", "EMBED-unique-mapping-mass-only-tfds", "INbreast-mass-only-tfds", 
                                    'BCDR-digital-mass-only-tfds', 'BCDR-film-mass-only-tfds',
                                    "CBIS-DDSM-calc-only-tfds", "EMBED-calc-only-tfds", "EMBED-unique-mapping-calc-only-tfds", "INbreast-calc-only-tfds", 
                                    'BCDR-digital-calc-only-tfds', 'BCDR-film-calc-only-tfds',
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
                                    
                                    "EMBED-unique-mapping-tfds-with-tabular-2classes",
                                    "EMBED-unique-mapping-tfds-with-tabular-demography-only-2classes"
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