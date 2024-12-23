CWD=$(pwd)
cd /home/hqvo2/Projects/MultiMEDal_multimodal_medical/src/test


# eva-clip base
python -m torch.distributed.launch --nproc_per_node=4 --master_port=5000 \
    test_timm.py --config-path /home/hqvo2/Projects/MultiMEDal_multimodal_medical/src/configs/paper_multimodal_config/cbis_ddsm/evaclip-vit-base-freeze_patches-224-tabular-ddsm_2classes_datasets.yaml

# eva-clip large
python -m torch.distributed.launch --nproc_per_node=4 --master_port=5000 \
    test_timm.py --config-path /home/hqvo2/Projects/MultiMEDal_multimodal_medical/src/configs/paper_multimodal_config/cbis_ddsm/evaclip-vit-large-freeze_patches-224-tabular-ddsm_2classes_datasets.yaml

# eva-clip giant
python -m torch.distributed.launch --nproc_per_node=4 --master_port=5000 \
    test_timm.py --config-path /home/hqvo2/Projects/MultiMEDal_multimodal_medical/src/configs/paper_multimodal_config/cbis_ddsm/evaclip-vit-giant-freeze_patches-224-tabular-ddsm_2classes_datasets.yaml


cd $CWD    