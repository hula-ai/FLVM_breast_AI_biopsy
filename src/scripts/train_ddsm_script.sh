CWD=$(pwd)
cd /home/hqvo2/Projects/MultiMEDal_multimodal_medical/src/train

# EVA-CLIP base
python -m torch.distributed.launch --nproc_per_node=4 --master_port=5008 \
    timm_train_concat_datasets.py --config-path /home/hqvo2/Projects/MultiMEDal_multimodal_medical/src/configs/patches_tabular_config/cbis_ddsm/evaclip_vit-base_patches-224-tabular-ddsm_2classes_datasets.yaml

# EVA-CLIP large
python -m torch.distributed.launch --nproc_per_node=4 --master_port=5008 \
    timm_train_concat_datasets.py --config-path /home/hqvo2/Projects/MultiMEDal_multimodal_medical/src/configs/patches_tabular_config/cbis_ddsm/evaclip_vit-large_patches-224-tabular-ddsm_2classes_datasets.yaml

# EVA-CLIP giant
python -m torch.distributed.launch --nproc_per_node=4 --master_port=5008 \
    timm_train_concat_datasets.py --config-path /home/hqvo2/Projects/MultiMEDal_multimodal_medical/src/configs/patches_tabular_config/cbis_ddsm/evaclip_vit-giant_patches-224-tabular-ddsm_2classes_datasets.yaml

cd $CWD