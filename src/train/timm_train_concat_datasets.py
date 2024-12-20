# %%
import torch
import timm
import os
import glob
import torch.nn as nn
import yaml
import argparse
import platform

from accelerate import DistributedDataParallelKwargs
from MultiMEDal_multimodal_medical.src.plotting.plot_funcs import (
    plot_pr_curve,
    plot_roc_curve,
)
from MultiMEDal_multimodal_medical.src.plotting.plot_funcs import (
    plot_pr_curve_crossval,
    plot_roc_curve_crossval,
)

from MultiMEDal_multimodal_medical.src.evaluation.compute_metrics import (
    compute_binary_metrics, compute_multilabel_metrics,
    compute_binary_metrics_crossval,
)
from MultiMEDal_multimodal_medical.src.evaluation.compute_metrics import (
    compute_multiclass_metrics,
    compute_multiclass_metrics_crossval,
)

from MultiMEDal_multimodal_medical.src.datasets.data_transform import (
    build_transform_dict,
    build_transform_dict_mamm,
    build_transform_dict_blip2,
    build_transform_dict_openclip
)
from MultiMEDal_multimodal_medical.src.datasets.data_loader import get_dataloaders
from MultiMEDal_multimodal_medical.src.datasets.dataset_init import get_datasets, get_combined_datasets
from MultiMEDal_multimodal_medical.src.datasets.custom_concat_dataset import CustomConcatDataset, CustomSubsetDataset
from MultiMEDal_multimodal_medical.src.datasets.preprocessing.prompt_factory import tab2prompt_breast_lesion
from MultiMEDal_multimodal_medical.src.test import test_utils

from MultiMEDal_multimodal_medical.src.datasets.custom_collate import custom_collate, collate_breast_cancer_patch_tabular

from MultiMEDal_multimodal_medical.src.models.open_clip import Clip_Image_Tabular_Model


from libauc.losses import AUCMLoss 
from libauc.optimizers import PESG 

from accelerate.utils import set_seed
from torch.utils.data import ConcatDataset
from itertools import chain




parser = argparse.ArgumentParser()

if platform.python_version() in ["3.10.0"]:
    parser.add_argument("--local-rank", type=int)  
elif platform.python_version() in ["3.7.16", "3.8.0", "3.8.18", "3.8.12"]:
    parser.add_argument("--local_rank", type=int)
else:
    raise ValueError("Need to add python version to check `local_rank` argument")

parser.add_argument("--config-path", dest="config_path")
parser.add_argument("--random-seed", default=None, type=int)
args = parser.parse_args()

with open(args.config_path, 'r') as file:
    yaml_cfg = yaml.safe_load(file)


# config_dict = {
#     "aim_repo": "/home/hqvo2/Projects/aim_experiments/breast_cancer_base_mamm",
#     "experiment_name": "ddsm-mamm-classifier",
#     "save_root": os.path.join("/home/hqvo2/Projects/MultiMEDal_multimodal_medical/experiments/resnet34_mamm_classifier", dt_string),
#     "dataset": "CBIS-DDSM-whole-mamm",
#     "data_dir": "CBIS_DDSM_pngs_448",
#     "image_size": (576, 448),
#     "model_name": "resnet34",
#     "pretrained": True,
#     "drop_rate": 0.7,
#     "drop_path_rate": 0.4,
#     "resume_ckpt": None,
#     "pretrain_ckpt": None,
#     "num_epochs": 10,
#     "warmup_eps": 1,
#     "njobs": 4,
#     "batch_size": 64,
#     "lr": 0.0001,
#     "use_weighted_classes": False,
#     "balanced_sampling": False,
#     "log_freq_eps": 1,
# }

import psutil, time
p = psutil.Process(os.getppid())
dt_string = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(p.create_time()))

config_dict = yaml_cfg['hyperparams']
config_dict['save_root'] = os.path.join(config_dict['save_root'], str(os.getppid()) + '_' + dt_string)

if config_dict.get('image_size') is not None:
    if isinstance(config_dict['image_size'], list):
        config_dict['image_size'] = tuple(config_dict['image_size'])


# %%
# image_size = 112
# image_size = (576, 448)
# image_size = (864, 672)
# image_size = (1152, 896)
mamm_datasets = ["CBIS-DDSM-whole-mamm", "CBIS-DDSM-whole-mamm-breast-density", "CBIS-DDSM-whole-mamm-abnormality", "EMBED-whole-mamm", "EMBED-whole-mamm-unique-mapping", "BMCD-whole-mamm", "INbreast-whole-mamm-train-only"]
patch_datasets = ["CBIS-DDSM-tfds", "EMBED-tfds", "EMBED-unique-mapping-tfds", "INbreast-tfds", 
                    'CBIS-DDSM-tfds-2classes', 'EMBED-tfds-2classes','EMBED-unique-mapping-tfds-2classes',
                    'BCDR-digital-tfds', 'BCDR-film-tfds',
                    "CBIS-DDSM-mass-only-tfds", "EMBED-mass-only-tfds", "EMBED-unique-mapping-mass-only-tfds", "INbreast-mass-only-tfds", 
                    'BCDR-digital-mass-only-tfds', 'BCDR-film-mass-only-tfds'
                    ]
tabular_datasets = ['CBIS-DDSM-tabular']
patch_tabular_datasets = ["CBIS-DDSM-tfds-with-tabular-2classes", "CBIS-DDSM-tfds-with-tabular-methodist-mass-appearance",
                        "CBIS-DDSM-tfds-with-tabular-methodist-calc-morph", "CBIS-DDSM-tfds-with-tabular-methodist-calc-dist",
                        "CBIS-DDSM-tfds-with-tabular-mass-shape", "CBIS-DDSM-tfds-with-tabular-mass-margin",
                        "CBIS-DDSM-tfds-with-tabular-calc-morph", "CBIS-DDSM-tfds-with-tabular-calc-dist",
                        "EMBED-unique-mapping-tfds-with-tabular-2classes", "EMBED-unique-mapping-tfds-with-tabular-demography-only-2classes",
                        "CBIS-DDSM-tfds-with-tabular-2classes-birad3",
                        "CBIS-DDSM-tfds-with-tabular-2classes-birad4",
                        "CBIS-DDSM-tfds-with-tabular-2classes-birad24",
                        "CBIS-DDSM-tfds-with-tabular-2classes-birad01245",
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
import torch.nn.functional as F
import numpy as np

# %%
import random
import numpy as np
import torch

# seed = 42
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.backends.cudnn.benchmark = False
# # torch.use_deterministic_algorithms(True)


# def seed_worker(worker_id):
#     worker_seed = torch.initial_seed() % 2**32
#     np.random.seed(worker_seed)
#     random.seed(worker_seed)


# g = torch.Generator()
# g.manual_seed(0)



# %%
dataset_name = config_dict['dataset']
data_dir = config_dict['datadir']
dataset_partition = config_dict['dataset_partition']

if isinstance(dataset_name, list):
    combined_datasets = get_combined_datasets(
        dataset_name[0],
        dataset_name[1],
        dataset_name[2],
        transform_dict,
        data_dir[0],
        data_dir[1],
        data_dir[2],
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
    train_patients = train_dataset.get_all_patients()


    # if len(set(config_dict["dataset"][0]).intersection(set(mamm_datasets))) != 0:
        
        
    # train_feats = list(chain(*[d.get_all_images() for d in all_train_datasets]))
    # train_labels = list(chain(*[d.get_all_labels() for d in all_train_datasets]))
    # train_patients = list(chain(*[d.get_all_patients() for d in all_train_datasets]))



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


# %%
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim
import transformers
import copy
import torchvision.models as torchvision_models

from torch.utils.data import (
    DataLoader,
    random_split,
    SubsetRandomSampler,
    ConcatDataset,
    WeightedRandomSampler,
)
from sklearn.utils import class_weight
from tqdm import tqdm
from sklearn.model_selection import KFold, StratifiedKFold
from accelerate import Accelerator
from accelerate import FullyShardedDataParallelPlugin
from accelerate.utils import ProjectConfiguration
from aim import Run

from torch.nn import Linear


from argparse import Namespace


def run_ann_pytorch(
    config_dict,
    train_dataset,
    test_dataset,
    val_dataset=None,
    train_idx=None,
    val_idx=None,
):

    cfg = Namespace(**config_dict)

    set_seed(cfg.seed)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=cfg.find_unused_parameters)

    if cfg.distributed_mode == 'FSDP':
        from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig
        fsdp_plugin = FullyShardedDataParallelPlugin(
            state_dict_config=FullStateDictConfig(offload_to_cpu=False, rank0_only=False),
            optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=False, rank0_only=False),
        )
        accelerator = Accelerator(mixed_precision=cfg.mixed_precision, fsdp_plugin=fsdp_plugin, project_dir=cfg.save_path, kwargs_handlers=[ddp_kwargs])
    elif cfg.distributed_mode == 'DDP':
        accelerator = Accelerator(mixed_precision=cfg.mixed_precision, project_dir=cfg.save_path, kwargs_handlers=[ddp_kwargs])
    else:
        raise ValueError("cfg.distributed mode must be in ['FSDP', 'DDP']!")


    n_classes = len(train_dataset.classes)
    data_type = cfg.data_type
    if accelerator.is_local_main_process:
        print("#classes:", n_classes)

    unique, counts = np.unique(train_dataset.get_all_labels(), return_counts=True)
    if accelerator.is_local_main_process:
        print("Training Set:", np.asarray((unique, counts)).T)

    if val_dataset is not None:
        unique, counts = np.unique(val_dataset.get_all_labels(), return_counts=True)
        
        if accelerator.is_local_main_process:
            print("Validation Set:", np.asarray((unique, counts)).T)

    unique, counts = np.unique(test_dataset.get_all_labels(), return_counts=True)
    if accelerator.is_local_main_process:
        print("Testing Set:", np.asarray((unique, counts)).T)

    if cfg.model_name in ['resnet18', 'resnet34', 'resnet50', 'resnet18d', 'resnet34d', 'resnet50d', "convnext_tiny.fb_in1k"]:
        model = timm.create_model(
            cfg.model_name,
            pretrained=cfg.pretrained,
            drop_rate=cfg.drop_rate,
            drop_path_rate=cfg.drop_path_rate,
            num_classes=n_classes,
        )
    elif cfg.model_name in ['ann']:        
        model = Ann(input_dim=train_dataset.get_feature_dim(), hidden_dim=cfg.hidden_dim, output_dim=n_classes, 
                    n_hidden=cfg.n_hidden, drop_rate=cfg.drop_rate)
    elif cfg.model_name in ['fusion_resnet34d']:
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

    if cfg.pretrain_ckpt is not None:
        state_dict = torch.load(cfg.pretrain_ckpt, map_location="cpu")
        for k in list(state_dict.keys()):
            # retain only base_encoder up to before the embedding layer
            if k.startswith("fc") or k.startswith("head.fc"):
                # remove prefix
                state_dict[k + ".old"] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]

        msg = model.load_state_dict(state_dict, strict=False)
        accelerator.print(msg)
    # model = create_model.mocov3()
    
    # if cfg.tune_bias_only:
    #     for name, param in model.named_parameters():
    #         if "bias" not in name:
    #             param.requires_grad = False"CBIS-DDSM-whole-mamm-breast-density"
    #             print("Freeze layer:", name)

    # Trainable Params
    accelerator.print("Trainable Params:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            accelerator.print(name, end=' ')
    print()

    device = accelerator.device

    train_sampler = None
    val_sampler = None
    if train_idx is not None and val_idx is not None:
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)

    train_labels = train_dataset.get_all_labels()

    if cfg.loss != "BCE":
        if train_idx is None:
            class_weights = class_weight.compute_class_weight(
                "balanced", classes=np.unique(train_labels), y=np.array(train_labels)
            )
        else:
            class_weights = class_weight.compute_class_weight(
                "balanced",
                classes=np.unique(train_labels),
                y=np.array(train_labels)[train_idx],
            )

    if cfg.use_weighted_classes:
        class_weights_dict = dict(zip(range(len(class_weights)), class_weights))
        accelerator.print("Classes weights:", class_weights_dict)
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights, device=device, dtype=torch.float)
        )
    else:
        if cfg.loss == 'cross-entropy':
            criterion = nn.CrossEntropyLoss()
        elif cfg.loss == 'AUCM':
            criterion = AUCMLoss()
        elif cfg.loss == 'BCE':
            criterion = nn.BCEWithLogitsLoss()

    if train_sampler is None and cfg.balanced_sampling:
        accelerator.print("[+] Balanced Oversampling", class_weights)
        sample_weights = [class_weights[i] for i in train_labels]
        train_sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(train_dataset), replacement=True
        )

    collate_fn = None
    if cfg.collate_fn == "breast_cancer_patch_tabular":
        collate_fn = collate_breast_cancer_patch_tabular

    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(
        train_dataset,
        test_dataset,
        val_dataset,
        train_sampler,
        val_sampler,
        cfg.batch_size,
        cfg.njobs,
        custom_collate_fn=collate_fn
    )

    model = accelerator.prepare(model)

    if cfg.opt == 'Adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=cfg.lr,
        )
    elif cfg.opt == 'PESG':
        optimizer = PESG(
            model.parameters(),
            loss_fn=criterion,
            lr=cfg.lr,
        )

    # LR scheduler
    total_samples = len(train_dataloader.dataset)
    num_warmup_steps = (total_samples // cfg.batch_size) * cfg.warmup_eps
    num_total_steps = (total_samples // cfg.batch_size) * cfg.num_epochs
    lr_scheduler = transformers.get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_total_steps
    )

    (
        # model,
        optimizer,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        # model,
        optimizer,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        lr_scheduler,
    )

    if cfg.resume_ckpt is not None:
        accelerator.load_state(cfg.resume_ckpt)

    if accelerator.is_local_main_process:
        run = Run(
            repo=cfg.aim_repo, experiment=cfg.experiment_name, log_system_params=True, capture_terminal_logs=True
        )
        run.description = config_dict['run_desc']
        run["hparams"] = config_dict

    # Best Model
    best_loss = float("inf")
    best_auroc = -float("inf")
    best_epoch = None

    for epoch in range(cfg.num_epochs):
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0

            if phase == "train":
                dataloader = train_dataloader
            elif phase == "val":
                dataloader = val_dataloader

            for step, batch_data in enumerate(dataloader):          
                if data_type == 'image':
                    samples, labels = batch_data["image"], batch_data["label"]
                elif data_type == 'tabular':
                    samples, labels = batch_data["feat_vec"], batch_data['label']
                elif data_type == 'image_tabular':
                    img_samples, feat_samples, labels = batch_data["image"], batch_data["feat_vec"], batch_data["label"]

                   
                    if cfg.tab_to_text:
                        # text_samples = []
                        # for breast_side, breast_density, abnormal, mass_shape, mass_margin, calc_morph, calc_dist in \
                        #     zip(batch_data["breast_side"], batch_data["breast_density"], \
                        #         batch_data["abnormal"], batch_data["mass_shape"], batch_data["mass_margin"], \
                        #         batch_data["calc_morph"], batch_data["calc_dist"]):                                                        

                        #     if abnormal == 'mass':
                        #         text =  f"A mass lesion with {mass_shape.lower()} shape and {mass_margin.lower()} margin " \
                        #                 f"is located in the {breast_side.lower()} breast. This {breast_side.lower()} has {breast_density.lower()} density."
                        #     elif abnormal == 'calcification':
                        #         text =  f"A calcification lesion with {calc_morph.lower()} appearance and {calc_dist.lower()} distribution " \
                        #                 f"is located in the {breast_side.lower()} breast. This {breast_side.lower()} has {breast_density.lower()} density."
                                
                        #     text_samples.append(text)

                        # if cfg.model_name in ['lavis_blip2', 'lavis_blip2_temp']:
                        #     txt_processor = txt_processors['train'] if phase == 'train' else txt_processors['eval']
                        #     text_samples = [txt_processor(text_sample) for text_sample in text_samples]
                        # elif cfg.model_name in ['open_clip']:
                        #     txt_processor = txt_tokenizer
                        #     text_samples = txt_processor(text_samples, context_length=256)

                        
                        text_samples, _ = tab2prompt_breast_lesion(cfg.model_name, phase,
                                                                batch_data, txt_processors, _context_length=CONTEXT_LENGTH, _group_age=cfg.group_age)
                            
                            


                # labels = labels.float().to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    if data_type in ['image', 'tabular']:
                        outputs = model(samples)
                    elif data_type in ['image_tabular']:
                        if cfg.tab_to_text:
                            outputs = model(img_samples, text_samples)
                        else:
                            outputs = model(img_samples, feat_samples)

                    if cfg.loss == 'AUCM':
                        outputs = torch.sigmoid(outputs)
                        
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        accelerator.backward(loss)
                        optimizer.step()
                        lr_scheduler.step()

                # statistics
                if data_type == 'image_tabular':
                    running_loss += loss.item() * img_samples.size(0)
                else:
                    running_loss += loss.item() * samples.size(0)

            epoch_loss = running_loss / len(dataloader.dataset)

            if accelerator.is_local_main_process:
                if epoch % cfg.log_freq_eps == 0:
                    print("Epoch:", epoch, f"{phase} Loss:", epoch_loss)
                run.track(epoch_loss, epoch=epoch, name=f"{phase} loss")

            split_preds_proba, _, split_labels = test_utils.predict_proba_pytorch(
                dataloader, model, 1, device, accelerator, data_type, cfg.tab_to_text, cfg.model_name, phase, txt_processors, CONTEXT_LENGTH, cfg.group_age
            )
            split_preds_proba = torch.from_numpy(split_preds_proba).to(device)
            split_labels = split_labels.to(device)

            if len(train_dataset.classes) == 2 and cfg.loss != "BCE":
                split_eval = compute_binary_metrics(
                    split_preds_proba, split_labels, device
                )
            else:
                if cfg.loss == "BCE":
                    split_eval = compute_multilabel_metrics(
                        split_preds_proba, split_labels, len(train_dataset.classes), device
                    )
                else:
                    split_eval = compute_multiclass_metrics(
                        split_preds_proba, split_labels, len(train_dataset.classes), device
                    )

            if accelerator.is_local_main_process:
                if epoch % cfg.log_freq_eps == 0:
                    print(
                        split_eval["acc"].item(),
                        split_eval["auroc"].item(),
                        split_eval["ap"].item(),
                    )

                run.track(split_eval["acc"].item(), epoch=epoch, name=f"{phase} acc")
                run.track(
                    split_eval["auroc"].item(), epoch=epoch, name=f"{phase} auroc"
                )
                run.track(split_eval["ap"].item(), epoch=epoch, name=f"{phase} ap")

            # deep copy the model
            if phase == "val":
                if epoch_loss < best_loss:
                    best_loss = epoch_loss

                if split_eval["auroc"].item() > best_auroc:
                    best_auroc = split_eval["auroc"].item()
                    best_epoch = epoch

                    output_dir = f"epoch_{epoch}"
                    # accelerator.wait_for_everyone()
                    accelerator.save_state(os.path.join(cfg.save_path, output_dir))

    # load best model weights
    accelerator.wait_for_everyone()  # wait until all distributed models finished saving
    accelerator.load_state(os.path.join(cfg.save_path, f"epoch_{best_epoch}"))

    accelerator.print("Best AUROC:", best_auroc, "Best Epoch:", best_epoch)

    model.eval()
    assert not model.training

    accelerator.print("# Test Samples:", len(test_dataloader.dataset))
    test_preds_proba, _, test_labels = test_utils.predict_proba_pytorch(
        test_dataloader, model, 1, device, accelerator, data_type, cfg.tab_to_text, cfg.model_name, 'test', txt_processors, CONTEXT_LENGTH, cfg.group_age
    )

    test_preds_proba = torch.from_numpy(test_preds_proba).to(device)
    test_labels = test_labels.to(device)

    # accelerator.print(test_preds_proba[:, 1])

    ### Test purpose
    # test_preds_proba_list = torch.argmax(test_preds_proba, axis=1).cpu().numpy().tolist()
    # test_list = test_labels.cpu().numpy().tolist()

    # test_preds_proba_unique, test_preds_proba_counts = np.unique(np.array(test_preds_proba_list), return_counts=True)
    # test_unique, test_counts = np.unique(np.array(test_list), return_counts=True)

    # accelerator.print(np.asarray((test_preds_proba_unique, test_preds_proba_counts)).T)
    # accelerator.print(np.asarray((test_unique, test_counts)).T)

    ### Test purpose

    if len(train_dataset.classes) == 2 and cfg.loss != "BCE":
        test_eval = compute_binary_metrics(test_preds_proba, test_labels, device)
    else:
        if cfg.loss == "BCE":
            test_eval = compute_multilabel_metrics(
                test_preds_proba, test_labels, len(test_dataset.classes), device
            )
        else:
            test_eval = compute_multiclass_metrics(
                test_preds_proba, test_labels, len(test_dataset.classes), device
            )


    accelerator.print(test_eval["acc"], test_eval["auroc"], test_eval["ap"])

    model = accelerator.unwrap_model(model)
    accelerator.save(model.state_dict(), os.path.join(cfg.save_path, "best_state.pkl"))
    accelerator.save(
        test_preds_proba, os.path.join(cfg.save_path, "test_preds_proba.pt")
    )
    accelerator.save(test_labels, os.path.join(cfg.save_path, "test_labels.pt"))

    # Free memory
    accelerator.clear()


if isinstance(dataset_name, str):
    if dataset_name != "CBIS-DDSM-tfds":
        for fold_id, (train_idx, val_idx) in enumerate(train_val_splits):
            save_path = os.path.join(config_dict["save_root"], f"fold_{fold_id}")
            os.makedirs(save_path, exist_ok=True)
            config_dict["save_path"] = save_path
            
            args = (config_dict, train_dataset, test_dataset, val_dataset, train_idx, val_idx)
            run_ann_pytorch(*args)        
    else:
        save_path = os.path.join(config_dict["save_root"], "fold_0")
        os.makedirs(save_path, exist_ok=True)

        config_dict["save_path"] = save_path
    
        args = (config_dict, train_dataset, test_dataset, val_dataset, None, None)
        run_ann_pytorch(*args)
elif isinstance(dataset_name, list):
    if config_dict["trial_runs"]["enable"]:
        run_desc = config_dict["run_desc"]
        # random_seeds = [42, 0, 1, 1234, 3407]
        if args.random_seed is not None:
            random_seeds = [args.random_seed]
            config_dict["trial_runs"]["random_seeds"] = random_seeds
        else:
            random_seeds = config_dict["trial_runs"]["random_seeds"]

        for seed in random_seeds:
            save_path = os.path.join(config_dict["save_root"], f"seed_{seed}")
            os.makedirs(save_path, exist_ok=True)

            config_dict["run_desc"] = f"{run_desc}_seed-{seed}"
            config_dict["seed"] = seed
            config_dict["save_path"] = save_path

            args = (config_dict, train_dataset, test_dataset, val_dataset)
            run_ann_pytorch(*args)
    else:
        save_path = os.path.join(config_dict["save_root"], "fold_0")
        os.makedirs(save_path, exist_ok=True)

        config_dict["save_path"] = save_path

        args = (config_dict, train_dataset, test_dataset, val_dataset)
        run_ann_pytorch(*args)
