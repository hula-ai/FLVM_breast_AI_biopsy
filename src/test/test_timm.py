# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import timm
import os
import glob
import yaml
import cv2
import re

import numpy as np
import transformers
import copy
import torchvision.models as torchvision_models
import matplotlib.pyplot as plt
import PIL
import aim
import argparse

from torch.utils.data import DataLoader, random_split, SubsetRandomSampler, ConcatDataset, WeightedRandomSampler
from sklearn.utils import class_weight
from tqdm import tqdm
from sklearn.model_selection import KFold, StratifiedKFold
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, broadcast
from aim import Run, Figure
from torch.nn import Linear
from argparse import Namespace

# %%

from MultiMEDal_multimodal_medical.src.models.open_clip import Clip_Image_Tabular_Model

from MultiMEDal_multimodal_medical.src.plotting.plot_funcs import plot_pr_curve, plot_roc_curve
from MultiMEDal_multimodal_medical.src.plotting.plot_funcs import plot_pr_curve_crossval, plot_roc_curve_crossval
from MultiMEDal_multimodal_medical.src.plotting.plot_funcs import plot_multiclass_roc_curve_crossval, plot_multiclass_pr_curve_crossval


from MultiMEDal_multimodal_medical.src.evaluation.compute_metrics import compute_binary_metrics_for_pr_crossval, compute_binary_metrics_for_roc_crossval
from MultiMEDal_multimodal_medical.src.evaluation.compute_metrics import compute_multiclass_metrics_for_pr_crossval, compute_multiclass_metrics_for_roc_crossval
from MultiMEDal_multimodal_medical.src.evaluation.compute_metrics import compute_multilabel_metrics_for_pr_crossval, compute_multilabel_metrics_for_roc_crossval
from MultiMEDal_multimodal_medical.src.evaluation.compute_metrics import compute_binary_metrics, compute_binary_metrics_crossval
from MultiMEDal_multimodal_medical.src.evaluation.compute_metrics import compute_multiclass_metrics, compute_multiclass_metrics_crossval
from MultiMEDal_multimodal_medical.src.evaluation.compute_metrics import compute_multilabel_metrics, compute_multilabel_metrics_crossval

from MultiMEDal_multimodal_medical.src.datasets.dataset_init import get_datasets, get_combined_datasets
from MultiMEDal_multimodal_medical.src.datasets.data_transform import (
    build_transform_dict,
    build_transform_dict_mamm,
    build_transform_dict_blip2,
    build_transform_dict_openclip
)
from MultiMEDal_multimodal_medical.src.datasets.custom_concat_dataset import CustomConcatDataset
from MultiMEDal_multimodal_medical.src.datasets.CBIS_DDSM import CBIS_DDSM_dataset, CBIS_DDSM_dataset_tfds, CBIS_DDSM_whole_mamm_dataset
from MultiMEDal_multimodal_medical.src.datasets.CBIS_DDSM import CBIS_DDSM_whole_mamm_breast_density_dataset

from MultiMEDal_multimodal_medical.src.datasets.data_transform import build_transform_dict, build_transform_dict_mamm
from MultiMEDal_multimodal_medical.src.datasets.custom_collate import custom_collate


from MultiMEDal_multimodal_medical.src.test import test_utils
from MultiMEDal_multimodal_medical.src.datasets.custom_collate import custom_collate, collate_breast_cancer_patch_tabular

# %%
from accelerate.utils import write_basic_config
from accelerate.utils import set_seed

# write_basic_config()  # Write a config file

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int)
parser.add_argument("--config-path", type=str)
args = parser.parse_args()




# %%
with open(args.config_path, 'r') as file:
    yaml_cfg = yaml.safe_load(file)

# %%
import psutil, time
p = psutil.Process(os.getppid())
dt_string = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(p.create_time()))

config_dict = yaml_cfg['hyperparams']
config_dict['save_root'] = os.path.join(config_dict['save_root'], str(os.getppid()) + '_' + dt_string)
if isinstance(config_dict['image_size'], list):
    config_dict['image_size'] = tuple(config_dict['image_size'])

# %%

def aim_track_fig_img(run, fig, fig_name, _context):
    aim_roc_figure = Figure(fig)
    run.track(aim_roc_figure, name=fig_name, step=0, context=_context)
    aim_roc_image = aim.Image(PIL.Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb()))
    run.track(value=aim_roc_image, name=f"{fig_name}_img", step=0, context=_context)
        

# %%
def test_ann_pytorch(config_dict, dataset, dataset_name: str, context_set, _run_hash=None, dataset_idx=None, log_freq_eps=10):   
    '''
    dataset_name - For logging purpose only
    '''
    cfg = Namespace(**config_dict)

    set_seed(42)
    accelerator = Accelerator(mixed_precision='fp16', project_dir=cfg.save_path)

    
    n_classes = len(dataset.classes)
    accelerator.print("#classes:", n_classes)

    data_sampler = None
    if dataset_idx is not None:
        data_sampler = SubsetRandomSampler(dataset_idx)

    collate_fn = None
    if cfg.collate_fn == "breast_cancer_patch_tabular":
        collate_fn = collate_breast_cancer_patch_tabular

    dataloader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=cfg.batch_size,
        num_workers=cfg.njobs,     
        sampler=data_sampler,   
        persistent_workers=True,
        collate_fn=collate_fn  
    )

    dataloader = accelerator.prepare(
        dataloader
    )

    if accelerator.is_local_main_process:
        if _run_hash is None:
            run = Run(
                repo=cfg.aim_repo, experiment=cfg.experiment_name, log_system_params=True, capture_terminal_logs=True
            )
            run.description = config_dict['run_desc']
            run["hparams"] = config_dict
        else:
            run = Run(repo=cfg.aim_repo, run_hash=_run_hash)

    if accelerator.is_local_main_process:
        run_hash = re.search(r"Run:\s(.*)", run.name).groups()[0]
    else:
        run_hash = None

    # broadcast(run_hash, 0)
    

    device = accelerator.device
    accelerator.print("#Samples:", len(dataloader.dataset))

    test_preds_proba_list = []
    test_labels_check = None

    if cfg.model_name in ['resnet18', 'resnet34', 'resnet50', 'resnet18d', 'resnet34d', 'resnet50d', \
                            "convnext_tiny.fb_in1k", 'eva_giant_patch14_clip_224.merged2b']:
        model = timm.create_model(
            cfg.model_name,
            pretrained=cfg.pretrained,
            drop_rate=cfg.drop_rate,
            drop_path_rate=cfg.drop_path_rate,
            num_classes=n_classes,
        )

    elif cfg.model_name in ['custom_resnet18d', 'custom_resnet34d', 'custom_resnet50d']:
        model_name = cfg.model_name.split('_')[1]
        model = Custom_Image_Model(
            model_name, 
            n_classes, 
            cfg.drop_rate, 
            cfg.drop_path_rate, 
            use_pretrained=cfg.pretrained)
    elif cfg.model_name in ['ann']:        
        model = Ann(input_dim=train_dataset.get_feature_dim(), hidden_dim=cfg.hidden_dim, output_dim=n_classes, 
                    n_hidden=cfg.n_hidden, drop_rate=cfg.drop_rate)
    elif cfg.model_name in ['fusion_resnet18d', 'fusion_resnet34d', 'fusion_resnet50d']:
        model = Image_Tabular_Concat_Model(model_name=cfg.img_model, input_vector_dim=train_dataset.get_feature_dim(),
                                           drop_rate=cfg.drop_rate, drop_path_rate=cfg.drop_path_rate,
                                           num_classes=n_classes, use_pretrained=cfg.pretrained)
    elif cfg.model_name in ['fusion_vit_giant', "fusion_vit_large", "fusion_vit_base"]:
        model = Image_Tabular_Concat_ViT_Model(model_name=cfg.img_model, input_vector_dim=train_dataset.get_feature_dim(),
                                           drop_rate=cfg.drop_rate, drop_path_rate=cfg.drop_path_rate,
                                           num_classes=n_classes, tab_encoder_layers=cfg.tab_encoder_layers,
                                           freeze_backbone=cfg.freeze_backbone, use_pretrained=cfg.pretrained)  
    elif cfg.model_name in ['fusion_crossatt_vit_giant', "fusion_crossatt_vit_large"]:
        model = Image_Tabular_CrossAtt_ViT_Model(model_name=cfg.img_model, input_vector_dim=train_dataset.get_feature_dim(),
                                           drop_rate=cfg.drop_rate, drop_path_rate=cfg.drop_path_rate,
                                           num_classes=n_classes, freeze_backbone=cfg.freeze_backbone, 
                                           crossatt_nhead=cfg.crossatt_nhead, crossatt_nlayer=cfg.crossatt_nlayer, use_pretrained=cfg.pretrained)                                                   
    elif cfg.model_name == 'lavis_blip2':        
        from MultiMEDal_multimodal_medical.src.models.blip2 import Blip2_Image_Tabular_Model
        from MultiMEDal_multimodal_medical.src.models.blip2_temp import Blip2_Image_Tabular_Temp_Model
        
        model = Blip2_Image_Tabular_Model(model_name=cfg.feat_ext_name, model_type=cfg.feat_ext_type, 
                                          num_classes=n_classes, drop_rate=cfg.drop_rate, 
                                          unfreeze_vis_encoder=cfg.unfreeze_vis_encoder, freeze_qformer=cfg.freeze_qformer, device=accelerator.device)
    elif cfg.model_name == 'open_clip':
        model = Clip_Image_Tabular_Model(model_name=cfg.pretrain_model_name, num_classes=n_classes, 
                                    drop_rate=cfg.drop_rate, freeze_backbone=cfg.freeze_backbone, pretrained_data=config_dict.get('pretrain_data', None))
    else:
        raise ValueError("Unrecognized Model Name!")

    for ckpt_id, ckpt_path in enumerate(cfg.ckpts_list):
        

        state_dict = torch.load(ckpt_path, map_location="cpu")

        msg = model.load_state_dict(state_dict, strict=False)
        accelerator.print(msg)

    
        model = accelerator.prepare(
            model
        )   
    
        model.eval()
        assert not model.training
    
        
        test_preds_proba, _, test_labels = test_utils.predict_proba_pytorch(
            dataloader, model, 1, device, accelerator, cfg.data_type, cfg.tab_to_text, 
            cfg.model_name, phase='test', txt_processor=txt_processors, context_length=CONTEXT_LENGTH,
            group_age=cfg.group_age
        )

        test_preds_proba = torch.from_numpy(test_preds_proba).to(device)
        test_labels = test_labels.to(device)        

        test_preds_proba_list.append(test_preds_proba)
        
        if test_labels_check is not None:
            assert np.array_equal(test_labels_check, test_labels), "test_labels must be the same for every iteration"

        model = accelerator.unwrap_model(model)

        os.makedirs(os.path.join(cfg.save_root, f"ckpt_{ckpt_id}"), exist_ok=True)

        accelerator.save(model.state_dict(), os.path.join(cfg.save_root, f"ckpt_{ckpt_id}", 'best_state.pkl'))
        accelerator.save(test_preds_proba, os.path.join(cfg.save_root, f"ckpt_{ckpt_id}", 'test_preds_proba.pt'))
        accelerator.save(test_labels, os.path.join(cfg.save_root, f"ckpt_{ckpt_id}", 'test_labels.pt')) 

        # clear memory every time loading a ckpt
        del state_dict
        torch.cuda.empty_cache()

    # Evaluate
    if len(test_labels.shape) == 1: # binary classes or multiclasses
        if n_classes == 2:        
            test_eval = compute_binary_metrics_crossval(test_preds_proba_list, test_labels, device)
        elif n_classes > 2:              
            test_eval = compute_multiclass_metrics_crossval(test_preds_proba_list, test_labels, n_classes, device)
    else: # multilabels
        test_eval = compute_multilabel_metrics_crossval(test_preds_proba_list, test_labels, n_classes, device)

    
    accelerator.print(f"Acc: {test_eval['acc']['mean']:.4f} \u00B1 {test_eval['acc']['std']:.4f};", 
                      f"AUC: {test_eval['auc']['mean']:.4f} \u00B1 {test_eval['auc']['std']:.4f};", 
                      f"AP: {test_eval['ap']['mean']:.4f} \u00B1 {test_eval['ap']['std']:.4f};")

    if accelerator.is_local_main_process:
        run.track({
            'Accuracy': test_eval['acc']['mean'],
            'AUROC': test_eval['auc']['mean'],
            'AP': test_eval['ap']['mean']
        }, context={**context_set, "moment": "mean"})
    
        run.track({
            'Accuracy': test_eval['acc']['std'],
            'AUROC': test_eval['auc']['std'],
            'AP': test_eval['ap']['std']
        }, context={**context_set, "moment": "std"})


    if accelerator.is_local_main_process:
        test_preds_proba_np_list = [test_preds_proba.cpu().numpy() for test_preds_proba in test_preds_proba_list]
        test_labels_np = test_labels.cpu().numpy()
        
        if len(test_labels.shape) == 1: # binary classes or multiclasses
            if n_classes == 2:                
                _compute_metrics_for_roc_crossval = compute_binary_metrics_for_roc_crossval
                _plot_roc_curve_crossval = plot_roc_curve_crossval

                _compute_metrics_for_pr_crossval = compute_binary_metrics_for_pr_crossval
                _plot_pr_curve_crossval = plot_pr_curve_crossval
                
            elif n_classes > 2:
                _compute_metrics_for_roc_crossval = compute_multiclass_metrics_for_roc_crossval
                _plot_roc_curve_crossval = plot_multiclass_roc_curve_crossval
        
                _compute_metrics_for_pr_crossval = compute_multiclass_metrics_for_pr_crossval
                _plot_pr_curve_crossval = plot_multiclass_pr_curve_crossval

        else: # multilabels
            _compute_metrics_for_roc_crossval = compute_multilabel_metrics_for_roc_crossval
            _plot_roc_curve_crossval = plot_multiclass_roc_curve_crossval

            _compute_metrics_for_pr_crossval = compute_multilabel_metrics_for_pr_crossval
            _plot_pr_curve_crossval = plot_multiclass_pr_curve_crossval


        # ROC curve track
        roc_crossval_metrics = _compute_metrics_for_roc_crossval(test_preds_proba_np_list, test_labels_np, device)
        roc_fig, roc_ax = plt.subplots()
        roc_fig = _plot_roc_curve_crossval(roc_fig, roc_ax, 'b', 'dimgray', *roc_crossval_metrics)
        # plt.show()
        # plt.close(roc_fig)
        aim_track_fig_img(run, roc_fig, fig_name="roc_curve", _context=context_set)

        # PR curve track
        pr_crossval_metrics = _compute_metrics_for_pr_crossval(test_preds_proba_np_list, test_labels_np, device)
        pr_fig, pr_ax = plt.subplots()
        pr_fig = _plot_pr_curve_crossval(pr_fig, pr_ax, 'b', *pr_crossval_metrics)

        aim_track_fig_img(run, pr_fig, fig_name="pr_curve", _context=context_set)

        
        
    # Free memory
    del model, dataloader
    accelerator.free_memory()
    

    return run_hash


    
    

mamm_datasets = ["CBIS-DDSM-whole-mamm", "CBIS-DDSM-whole-mamm-breast-density", "CBIS-DDSM-whole-mamm-abnormality"]
patch_datasets = ["CBIS-DDSM-tfds",
                    'CBIS-DDSM-tfds-2classes',
                    "CBIS-DDSM-mass-only-tfds"
                ]
tabular_datasets = ['CBIS-DDSM-tabular']
patch_tabular_datasets = ["CBIS-DDSM-tfds-with-tabular-2classes", "CBIS-DDSM-tfds-with-tabular-methodist-mass-appearance",
                        "CBIS-DDSM-tfds-with-tabular-methodist-calc-morph", "CBIS-DDSM-tfds-with-tabular-methodist-calc-dist",
                        "CBIS-DDSM-tfds-with-tabular-mass-shape", "CBIS-DDSM-tfds-with-tabular-mass-margin",
                        "CBIS-DDSM-tfds-with-tabular-calc-morph", "CBIS-DDSM-tfds-with-tabular-calc-dist",                          
                          "CBIS-DDSM-tfds-with-tabular-2classes-birad3",
                          "CBIS-DDSM-tfds-with-tabular-2classes-birad4",
                          ]


CONTEXT_LENGTH = None
if config_dict.get('model_name') in ['lavis_blip2', 'lavis_blip2_temp']:
    if config_dict.get('transform') == 'blip2_transform':
        transform_dict, txt_processors = build_transform_dict_blip2(config_dict.get('feat_ext_name'), config_dict.get('feat_ext_type'), input_size=config_dict["image_size"])
    elif config_dict.get('transform') == 'lesion_default':
        blip2_transform_dict, txt_processors = build_transform_dict_blip2(config_dict.get('feat_ext_name'), config_dict.get('feat_ext_type'), input_size=config_dict["image_size"])
        transform_dict = build_transform_dict(input_size=config_dict['image_size'],
                                              norm_mean=blip2_transform_dict['train'].transforms[-1].mean,
                                              norm_std=blip2_transform_dict['train'].transforms[-1].std)

elif config_dict.get('model_name') in ['open_clip']:
    
    transform_dict, txt_processors = build_transform_dict_openclip(config_dict.get('pretrain_model_name'), 
                                                                pretrained_data=config_dict.get('pretrain_data', None))
    if config_dict.get('pretrain_model_name') == "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224":
        CONTEXT_LENGTH = 256                                             
else:
    if len(set(config_dict["dataset"][0]).intersection(set(mamm_datasets))) != 0 :
        transform_dict = build_transform_dict_mamm(input_size=config_dict["image_size"])
    elif len(set(config_dict["dataset"][0]).intersection(set(patch_datasets))) != 0:
        transform_dict = build_transform_dict(input_size=config_dict['image_size'])
    elif len(set(config_dict["dataset"][0]).intersection(set(tabular_datasets))) != 0:
        transform_dict = None
    elif len(set(config_dict["dataset"][0]).intersection(set(patch_tabular_datasets))) != 0:
        transform_dict = build_transform_dict(input_size=config_dict['image_size'])

    txt_processors = None


# %%
dataset_names = config_dict['dataset']
data_dirs = config_dict['datadir']
dataset_partition = config_dict['dataset_partition']

combined_datasets = get_combined_datasets(
        dataset_names[0],
        dataset_names[1],
        dataset_names[2],
        transform_dict,
        data_dirs[0],
        data_dirs[1],
        data_dirs[2],
        dataset_partition[0],
        dataset_partition[1],
        dataset_partition[2]
)
all_train_datasets, all_val_datasets, all_test_datasets = combined_datasets
train_dataset = CustomConcatDataset(all_train_datasets)
val_dataset = CustomConcatDataset(all_val_datasets)
test_dataset = CustomConcatDataset(all_test_datasets)

train_feats = train_dataset.get_all_images()
train_labels = train_dataset.get_all_labels()
if config_dict['kfolds_train_split']['enable']: # for stratified group k-folds
    train_patients = train_dataset.get_all_patients()



# %%
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedGroupKFold
from MultiMEDal_multimodal_medical.src.datasets.custom_concat_dataset import CustomSubsetDataset

# if not isinstance(dataset_name, list):
if config_dict['kfolds_train_split']['enable']:
    num_folds = config_dict['kfolds_train_split']['k_folds']
    kfolds_rand_seed = config_dict['kfolds_train_split']['seed']
    
    splits = StratifiedGroupKFold(n_splits=num_folds, shuffle=True, random_state=kfolds_rand_seed)

    train_val_splits = []
    for fold, (train_idx, val_idx) in enumerate(
        splits.split(train_feats, train_labels, train_patients)
    ):
        train_val_splits.append((train_idx, val_idx))


    if config_dict['use_more_data_for_validation']:
        train_train_dataset = CustomSubsetDataset(train_dataset, train_val_splits[0][1]) # use val split for training
        train_val_dataset = CustomSubsetDataset(train_dataset, train_val_splits[0][0]) # use train split for additional validation

        train_dataset = train_train_dataset
        val_dataset = CustomConcatDataset([train_val_dataset, val_dataset])
    else:
        train_train_dataset = CustomSubsetDataset(train_dataset, train_val_splits[0][0]) # use train split as is
        train_val_dataset = CustomSubsetDataset(train_dataset, train_val_splits[0][1]) # use val split as is
        
        train_dataset = train_train_dataset
        val_dataset = CustomConcatDataset([train_val_dataset, val_dataset])



datasets = [train_dataset, val_dataset, test_dataset]

dataset_idx = None

ckpts_list = config_dict["ckpts_list"]

run_desc = config_dict['run_desc']

config_dict["save_path"] = []

context_list = [{"subset": "train"}, {"subset": "val"}, {"subset": "test"}]



run_hash = None
for dataset_name, data_dir, dataset, context_name in zip(dataset_names, data_dirs, datasets, context_list):


    args = (config_dict, dataset, dataset_name, context_name, run_hash, dataset_idx, 10)
    run_hash = test_ann_pytorch(*args)
