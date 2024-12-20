CWD=$(pwd)
cd /home/hqvo2/Projects/MultiMEDal_multimodal_medical/src/test


# eva-clip base (birads3)
python -m torch.distributed.launch --nproc_per_node=4 --master_port=5000 \
    test_timm.py --config-path /home/hqvo2/Projects/MultiMEDal_multimodal_medical/src/configs/paper_multimodal_config/patches_tabular_config_final/cbis_ddsm_birads3/evaclip-vit-base-freeze_patches-224-tabular-ddsm_2classes_hard_val_datasets.yaml

# eva-clip large (birads3)
python -m torch.distributed.launch --nproc_per_node=4 --master_port=5000 \
    test_timm.py --config-path /home/hqvo2/Projects/MultiMEDal_multimodal_medical/src/configs/paper_multimodal_config/patches_tabular_config_final/cbis_ddsm_birads3/evaclip-vit-large-freeze_patches-224-tabular-ddsm_2classes_hard_val_datasets.yaml

# eva-clip giant (birads3)
python -m torch.distributed.launch --nproc_per_node=4 --master_port=5000 \
    test_timm.py --config-path /home/hqvo2/Projects/MultiMEDal_multimodal_medical/src/configs/paper_multimodal_config/patches_tabular_config_final/cbis_ddsm_birads3/evaclip-vit-giant-freeze_patches-224-tabular-ddsm_2classes_hard_val_datasets.yaml


cd $CWD    