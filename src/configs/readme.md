* Config files are structured in two sub-directories as below. One is used for training and the other is for testing
* For training configuration files (configuration files located in the `patches_tabular_config` directory), there are two hyperameters need to be changed `aim_repo` (for logging path) and `save_root` (for checkpoint path)
* For testing configuration files (configuration files located in `paper_multimodal_config` directory), beside `aim_repo` and `save_root`, there is also a hyper-parameter `ckpts_list` need changing. This is a list of pathes to all saved checkpoints.

```sh
.
    # FOR TESTING
├── paper_multimodal_config
        # All BI-RADS cases
│   ├── cbis_ddsm
            # Vision Transformer Base architecture
│   │   ├── evaclip-vit-base-freeze_patches-224-tabular-ddsm_2classes_datasets.yaml
            # Vision Transformer Giant architecture
│   │   ├── evaclip-vit-giant-freeze_patches-224-tabular-ddsm_2classes_datasets.yaml
            # Vision Transformer Large architecture
│   │   └── evaclip-vit-large-freeze_patches-224-tabular-ddsm_2classes_datasets.yaml
        # Only BI-RADS 3 cases
│   └── cbis_ddsm_birads3
            # Vision Transformer Base architecture
│       ├── evaclip-vit-base-freeze_patches-224-tabular-ddsm_2classes_datasets.yaml
            # Vision Transformer Base architecture - using harder validation set
│       ├── evaclip-vit-base-freeze_patches-224-tabular-ddsm_2classes_hard_val_datasets.yaml
            # Vision Transformer Giant architecture
│       ├── evaclip-vit-giant-freeze_patches-224-tabular-ddsm_2classes_datasets.yaml
            # Vision Transformer Giant architecture - using harder validation set
│       ├── evaclip-vit-giant-freeze_patches-224-tabular-ddsm_2classes_hard_val_datasets.yaml
            # Vision Transformer Large architecture
│       ├── evaclip-vit-large-freeze_patches-224-tabular-ddsm_2classes_datasets.yaml
            # Vision Transformer Large architecture - using harder validation set
│       └── evaclip-vit-large-freeze_patches-224-tabular-ddsm_2classes_hard_val_datasets.yaml
    # FOR TRAINING
├── patches_tabular_config
        # All BI-RADS cases
│   ├── cbis_ddsm
            # Vision Transformer Base architecture
│   │   ├── evaclip_vit-base_patches-224-tabular-ddsm_2classes_datasets.yaml
            # Vision Transformer Giant architecture
│   │   ├── evaclip_vit-giant_patches-224-tabular-ddsm_2classes_datasets.yaml
            # Vision Transformer Large architecture
│   │   └── evaclip_vit-large_patches-224-tabular-ddsm_2classes_datasets.yaml
        # Only BI-RADS 3 cases
│   └── cbis_ddsm_birads3
            # Vision Transformer Base architecture
│       ├── evaclip_vit-base_patches-224-tabular-ddsm_2classes_datasets.yaml
            # Vision Transformer Base architecture - using harder validation set
│       ├── evaclip_vit-base_patches-224-tabular-ddsm_2classes_hard_val_datasets.yaml
            # Vision Transformer Giant architecture
│       ├── evaclip_vit-giant_patches-224-tabular-ddsm_2classes_datasets.yaml
            # Vision Transformer Giant architecture - using harder validation set
│       ├── evaclip_vit-giant_patches-224-tabular-ddsm_2classes_hard_val_datasets.yaml
            # Vision Transformer Large architecture
│       ├── evaclip_vit-large_patches-224-tabular-ddsm_2classes_datasets.yaml
            # Vision Transformer Large architecture - using harder validation set
│       └── evaclip_vit-large_patches-224-tabular-ddsm_2classes_hard_val_datasets.yaml
```