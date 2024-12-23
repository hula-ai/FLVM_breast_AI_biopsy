CWD=$(pwd)
cd /home/hqvo2/Projects/MultiMEDal_multimodal_medical/src/train


# Normal Split
python -m torch.distributed.launch --nproc_per_node=4 --master_port=6006 \
    timm_train_concat_datasets.py \
    --config-path /home/hqvo2/Projects/MultiMEDal_multimodal_medical/src/configs/patches_tabular_config/cbis_ddsm_birads3/evaclip_vit-base_patches-224-tabular-ddsm_2classes_datasets.yaml

python -m torch.distributed.launch --nproc_per_node=4 --master_port=6006 \
    timm_train_concat_datasets.py \
    --config-path /home/hqvo2/Projects/MultiMEDal_multimodal_medical/src/configs/patches_tabular_config/cbis_ddsm_birads3/evaclip_vit-large_patches-224-tabular-ddsm_2classes_datasets.yaml

python -m torch.distributed.launch --nproc_per_node=4 --master_port=6006 \
    timm_train_concat_datasets.py \
    --config-path /home/hqvo2/Projects/MultiMEDal_multimodal_medical/src/configs/patches_tabular_config/cbis_ddsm_birads3/evaclip_vit-giant_patches-224-tabular-ddsm_2classes_datasets.yaml

# Hard Validation
python -m torch.distributed.launch --nproc_per_node=4 --master_port=6006 \
    timm_train_concat_datasets.py \
    --config-path /home/hqvo2/Projects/MultiMEDal_multimodal_medical/src/configs/patches_tabular_config/cbis_ddsm_birads3/evaclip_vit-base_patches-224-tabular-ddsm_2classes_hard_val_datasets.yaml

python -m torch.distributed.launch --nproc_per_node=4 --master_port=6006 \
    timm_train_concat_datasets.py \
    --config-path /home/hqvo2/Projects/MultiMEDal_multimodal_medical/src/configs/patches_tabular_config/cbis_ddsm_birads3/evaclip_vit-large_patches-224-tabular-ddsm_2classes_hard_val_datasets.yaml

python -m torch.distributed.launch --nproc_per_node=4 --master_port=6006 \
    timm_train_concat_datasets.py \
    --config-path /home/hqvo2/Projects/MultiMEDal_multimodal_medical/src/configs/patches_tabular_config/cbis_ddsm_birads3/evaclip_vit-giant_patches-224-tabular-ddsm_2classes_hard_val_datasets.yaml


cd $CWD