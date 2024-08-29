import torch
import timm
import os
import glob
import torch.nn as nn
import yaml
import argparse
import platform
import pickle
import psutil, time
import torch.nn.functional as F
import numpy as np
import random
import torch

import torch.optim as optim
import transformers
import copy
import torchvision.models as torchvision_models

from argparse import Namespace
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedGroupKFold

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
from MultiMEDal_multimodal_medical.src.models import create_model
from torch.nn import Linear

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
    compute_binary_metrics,
    compute_multilabel_metrics,
    compute_binary_metrics_crossval,
)
from MultiMEDal_multimodal_medical.src.evaluation.compute_metrics import (
    compute_multiclass_metrics,
    compute_multiclass_metrics_crossval,
)

from MultiMEDal_multimodal_medical.src.datasets.data_transform import (
    build_transform_dict,
)

from MultiMEDal_multimodal_medical.src.datasets.data_loader import get_dataloaders
from MultiMEDal_multimodal_medical.src.datasets.dataset_init import (
    get_datasets,
    get_combined_datasets,
)
from MultiMEDal_multimodal_medical.src.datasets.custom_concat_dataset import (
    CustomConcatDataset,
)
from MultiMEDal_multimodal_medical.src.datasets.preprocessing.prompt_factory import (
    tab2prompt_breast_lesion,
)
from MultiMEDal_multimodal_medical.src.test import test_utils

from MultiMEDal_multimodal_medical.src.models.image_tabular_net import (
    Image_Tabular_Concat_ViT_Model,
)

from MultiMEDal_multimodal_medical.src.models.fusion_clip_model import (
    Clip_Image_Tabular_Model,
)


from accelerate.utils import set_seed
from torch.utils.data import ConcatDataset
from itertools import chain


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
    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=cfg.find_unused_parameters
    )

    if cfg.distributed_mode == "FSDP":
        from torch.distributed.fsdp.fully_sharded_data_parallel import (
            FullOptimStateDictConfig,
            FullStateDictConfig,
        )

        fsdp_plugin = FullyShardedDataParallelPlugin(
            state_dict_config=FullStateDictConfig(
                offload_to_cpu=False, rank0_only=False
            ),
            optim_state_dict_config=FullOptimStateDictConfig(
                offload_to_cpu=False, rank0_only=False
            ),
        )
        accelerator = Accelerator(
            mixed_precision=cfg.mixed_precision,
            fsdp_plugin=fsdp_plugin,
            project_dir=cfg.save_path,
            kwargs_handlers=[ddp_kwargs],
        )
    elif cfg.distributed_mode == "DDP":
        accelerator = Accelerator(
            mixed_precision=cfg.mixed_precision,
            project_dir=cfg.save_path,
            kwargs_handlers=[ddp_kwargs],
        )
    else:
        raise ValueError("cfg.distributed mode must be in ['FSDP', 'DDP']!")

    n_classes = len(train_dataset.classes)
    data_type = cfg.data_type
    accelerator.print("#classes:", n_classes)

    if cfg.model_name in ["fusion_vit_giant", "fusion_vit_large", "fusion_vit_base"]:
        model = Image_Tabular_Concat_ViT_Model(
            model_name=cfg.img_model,
            input_vector_dim=train_dataset.get_feature_dim(),
            drop_rate=cfg.drop_rate,
            drop_path_rate=cfg.drop_path_rate,
            num_classes=n_classes,
            freeze_backbone=cfg.freeze_backbone,
            use_pretrained=cfg.pretrained,
        )
    elif cfg.model_name in ["open_clip", "pubmed_clip"]:
        model = Clip_Image_Tabular_Model(
            model_name=cfg.pretrain_model_name,
            num_classes=n_classes,
            drop_rate=cfg.drop_rate,
            freeze_backbone=cfg.freeze_backbone,
            pretrained_data=config_dict.get("pretrain_data", None),
        )

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

    # Trainable Params
    accelerator.print("Trainable Params:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            accelerator.print(name, end=" ")

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
        if cfg.loss == "cross-entropy":
            criterion = nn.CrossEntropyLoss()
        elif cfg.loss == "BCE":
            criterion = nn.BCEWithLogitsLoss()

    if train_sampler is None and cfg.balanced_sampling:
        accelerator.print("[+] Balanced Oversampling", class_weights)
        sample_weights = [class_weights[i] for i in train_labels]
        train_sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(train_dataset), replacement=True
        )

    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(
        train_dataset,
        test_dataset,
        val_dataset,
        train_sampler,
        val_sampler,
        cfg.batch_size,
        cfg.njobs,
    )

    model = accelerator.prepare(model)

    optimizer = optim.Adam(
        model.parameters(),
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
        optimizer,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
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
            repo=cfg.aim_repo,
            experiment=cfg.experiment_name,
            log_system_params=True,
            capture_terminal_logs=True,
        )
        run.description = config_dict["run_desc"]
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
                if data_type == "image":
                    samples, labels = batch_data["image"], batch_data["label"]
                elif data_type == "tabular":
                    samples, labels = batch_data["feat_vec"], batch_data["label"]
                elif data_type == "image_tabular":
                    img_samples, feat_samples, labels = (
                        batch_data["image"],
                        batch_data["feat_vec"],
                        batch_data["label"],
                    )

                    if cfg.tab_to_text:
                        text_samples, raw_text_samples = tab2prompt_breast_lesion(
                            cfg.model_name,
                            phase,
                            batch_data,
                            txt_processors,
                            _context_length=CONTEXT_LENGTH,
                            _group_age=cfg.group_age,
                        )

                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    if data_type in ["image", "tabular"]:
                        outputs = model(samples)
                    elif data_type in ["image_tabular"]:
                        if cfg.tab_to_text:
                            if cfg.model_name == "pubmed_clip":
                                outputs = model(
                                    img_samples, text_samples[:, :77]
                                )  # maximum seq len for positional embeddings
                            else:
                                outputs = model(img_samples, text_samples)
                        else:
                            outputs = model(img_samples, feat_samples)

                    if cfg.loss == "AUCM":
                        outputs = torch.sigmoid(outputs)

                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        accelerator.backward(loss)
                        optimizer.step()
                        lr_scheduler.step()

                # statistics
                if data_type == "image_tabular":
                    running_loss += loss.item() * img_samples.size(0)
                else:
                    running_loss += loss.item() * samples.size(0)

            epoch_loss = running_loss / len(dataloader.dataset)

            if accelerator.is_local_main_process:
                if epoch % cfg.log_freq_eps == 0:
                    print("Epoch:", epoch, f"{phase} Loss:", epoch_loss)
                run.track(epoch_loss, epoch=epoch, name=f"{phase} loss")

            split_preds_proba, _, split_labels = test_utils.predict_proba_pytorch(
                dataloader,
                model,
                1,
                device,
                accelerator,
                data_type,
                cfg.tab_to_text,
                cfg.model_name,
                phase,
                txt_processors,
                CONTEXT_LENGTH,
                cfg.group_age,
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
                        split_preds_proba,
                        split_labels,
                        len(train_dataset.classes),
                        device,
                    )
                else:
                    split_eval = compute_multiclass_metrics(
                        split_preds_proba,
                        split_labels,
                        len(train_dataset.classes),
                        device,
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
                    accelerator.save_state(os.path.join(cfg.save_path, output_dir))

    # load best model weights
    accelerator.wait_for_everyone()  # wait until all distributed models finished saving
    accelerator.load_state(os.path.join(cfg.save_path, f"epoch_{best_epoch}"))

    accelerator.print("Best AUROC:", best_auroc, "Best Epoch:", best_epoch)

    model.eval()
    assert not model.training

    accelerator.print("# Test Samples:", len(test_dataloader.dataset))
    test_preds_proba, _, test_labels = test_utils.predict_proba_pytorch(
        test_dataloader,
        model,
        1,
        device,
        accelerator,
        data_type,
        cfg.tab_to_text,
        cfg.model_name,
        "test",
        txt_processors,
        CONTEXT_LENGTH,
        cfg.group_age,
    )

    test_preds_proba = torch.from_numpy(test_preds_proba).to(device)
    test_labels = test_labels.to(device)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config-path", dest="config_path")
    args = parser.parse_args()

    with open(args.config_path, "r") as file:
        yaml_cfg = yaml.safe_load(file)

    p = psutil.Process(os.getppid())
    dt_string = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(p.create_time()))

    config_dict = yaml_cfg["hyperparams"]
    config_dict["save_root"] = os.path.join(
        config_dict["save_root"], str(os.getppid()) + "_" + dt_string
    )

    if config_dict.get("image_size") is not None:
        if isinstance(config_dict["image_size"], list):
            config_dict["image_size"] = tuple(config_dict["image_size"])

    patch_tabular_datasets = [
        "CBIS-DDSM-tfds-with-tabular-2classes"
        "EMBED-unique-mapping-tfds-with-tabular-2classes"
    ]

    CONTEXT_LENGTH = None

    transform_dict = build_transform_dict(input_size=config_dict["image_size"])
    txt_processors = None

    dataset_name = config_dict["dataset"]
    data_dir = config_dict["datadir"]

    if isinstance(dataset_name, list):
        combined_datasets = get_combined_datasets(
            dataset_name[0],
            dataset_name[1],
            dataset_name[2],
            transform_dict,
            data_dir[0],
            data_dir[1],
            data_dir[2],
        )

        all_train_datasets, all_val_datasets, all_test_datasets = combined_datasets
        train_dataset = CustomConcatDataset(all_train_datasets)
        val_dataset = CustomConcatDataset(all_val_datasets)
        test_dataset = CustomConcatDataset(all_test_datasets)

        train_labels = train_dataset.get_all_labels()
        if len(set(config_dict["dataset"][0]).intersection(set(mamm_datasets))) != 0:
            train_feats = train_dataset.get_all_images()
            train_patients = train_dataset.get_all_patients()

    unique, counts = np.unique(train_dataset.get_all_labels(), return_counts=True)
    print(np.asarray((unique, counts)).T)

    if val_dataset is not None:
        unique, counts = np.unique(val_dataset.get_all_labels(), return_counts=True)
        print(np.asarray((unique, counts)).T)

    unique, counts = np.unique(test_dataset.get_all_labels(), return_counts=True)
    print(np.asarray((unique, counts)).T)

    # Run Training
    if config_dict["trial_runs"]["enable"]:
        run_desc = config_dict["run_desc"]

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
