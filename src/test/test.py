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
import pickle

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
from accelerate.utils import ProjectConfiguration, broadcast
from aim import Run, Figure
from torch.nn import Linear
from argparse import Namespace


from MultiMEDal_multimodal_medical.src.models import create_model
from MultiMEDal_multimodal_medical.src.models.image_tabular_net import (
    Image_Tabular_Concat_ViT_Model,
)

from MultiMEDal_multimodal_medical.src.models.fusion_clip_model import (
    Clip_Image_Tabular_Model,
)

from MultiMEDal_multimodal_medical.src.plotting.plot_funcs import (
    plot_pr_curve,
    plot_roc_curve,
)
from MultiMEDal_multimodal_medical.src.plotting.plot_funcs import (
    plot_pr_curve_crossval,
    plot_roc_curve_crossval,
)
from MultiMEDal_multimodal_medical.src.plotting.plot_funcs import (
    plot_multiclass_roc_curve_crossval,
    plot_multiclass_pr_curve_crossval,
)


from MultiMEDal_multimodal_medical.src.evaluation.compute_metrics import (
    compute_binary_metrics_for_pr_crossval,
    compute_binary_metrics_for_roc_crossval,
)
from MultiMEDal_multimodal_medical.src.evaluation.compute_metrics import (
    compute_multiclass_metrics_for_pr_crossval,
    compute_multiclass_metrics_for_roc_crossval,
)
from MultiMEDal_multimodal_medical.src.evaluation.compute_metrics import (
    compute_multilabel_metrics_for_pr_crossval,
    compute_multilabel_metrics_for_roc_crossval,
)
from MultiMEDal_multimodal_medical.src.evaluation.compute_metrics import (
    compute_binary_metrics,
    compute_binary_metrics_crossval,
)
from MultiMEDal_multimodal_medical.src.evaluation.compute_metrics import (
    compute_multiclass_metrics,
    compute_multiclass_metrics_crossval,
)
from MultiMEDal_multimodal_medical.src.evaluation.compute_metrics import (
    compute_multilabel_metrics,
    compute_multilabel_metrics_crossval,
)

from MultiMEDal_multimodal_medical.src.datasets.dataset_init import (
    get_datasets,
    get_combined_datasets,
)
from MultiMEDal_multimodal_medical.src.datasets.data_transform import (
    build_transform_dict,
)

from MultiMEDal_multimodal_medical.src.datasets.custom_concat_dataset import (
    CustomConcatDataset,
)
from MultiMEDal_multimodal_medical.src.datasets.CBIS_DDSM import (
    CBIS_DDSM_dataset,
    CBIS_DDSM_dataset_tfds,
)
from MultiMEDal_multimodal_medical.src.datasets.EMBED import (
    EMBED_dataset,
    EMBED_dataset_tfds,
)
from MultiMEDal_multimodal_medical.src.datasets.data_transform import (
    build_transform_dict,
)
from MultiMEDal_multimodal_medical.src.datasets.custom_collate import custom_collate


from MultiMEDal_multimodal_medical.src.test import test_utils
from accelerate.utils import write_basic_config
from accelerate.utils import set_seed


def aim_track_fig_img(run, fig, fig_name, _context):
    aim_roc_figure = Figure(fig)
    run.track(aim_roc_figure, name=fig_name, step=0, context=_context)
    aim_roc_image = aim.Image(
        PIL.Image.frombytes(
            "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
        )
    )
    run.track(value=aim_roc_image, name=f"{fig_name}_img", step=0, context=_context)


def test_ann_pytorch(
    config_dict,
    dataset,
    dataset_name: str,
    context_set,
    _run_hash=None,
    dataset_idx=None,
    log_freq_eps=10,
    feature_dim=None,
):
    """
    dataset_name - For logging purpose only
    _run_hash - For logging result and plotting figures in a specific Aim Run with run hash
    """
    cfg = Namespace(**config_dict)

    set_seed(42)
    accelerator = Accelerator(mixed_precision="fp16", project_dir=cfg.save_path)

    if cfg.combine_crossval is True:
        n_classes = len(dataset[0].classes)
    else:
        n_classes = len(dataset.classes)

    accelerator.print("#classes:", n_classes)

    data_sampler = None
    if dataset_idx is not None:
        data_sampler = SubsetRandomSampler(dataset_idx)

    if cfg.combine_crossval is True:
        assert (
            (cfg.combine_crossval == True)
            and (data_sampler is None)
            and isinstance(dataset, list)
        )
        assert len(cfg.ckpts_list) == len(
            dataset
        ), f" Variable cfg.ckpts_list has {len(cfg.ckpts_list)} checkpoints. Variable `dataset` has {len(dataset)} datasets."

        all_dataloaders = []
        for _ds in dataset:
            dataloader = DataLoader(
                _ds,
                shuffle=False,
                batch_size=cfg.batch_size,
                num_workers=cfg.njobs,
                sampler=data_sampler,
                persistent_workers=True,
            )

            dataloader = accelerator.prepare(dataloader)

            all_dataloaders.append(dataloader)
    else:
        dataloader = DataLoader(
            dataset,
            shuffle=False,
            batch_size=cfg.batch_size,
            num_workers=cfg.njobs,
            sampler=data_sampler,
            persistent_workers=True,
        )

        dataloader = accelerator.prepare(dataloader)

    if accelerator.is_local_main_process:
        if _run_hash is None:
            run = Run(
                repo=cfg.aim_repo,
                experiment=cfg.experiment_name,
                log_system_params=True,
                capture_terminal_logs=True,
            )
            run.description = config_dict["run_desc"]
            run["hparams"] = config_dict
        else:
            run = Run(repo=cfg.aim_repo, run_hash=_run_hash)

    if accelerator.is_local_main_process:
        run_hash = re.search(r"Run:\s(.*)", run.name).groups()[0]
    else:
        run_hash = None

    device = accelerator.device
    accelerator.print("#Samples:", len(dataloader.dataset))

    test_preds_proba_list = []
    test_labels_check = None

    if cfg.model_name in ["fusion_vit_giant", "fusion_vit_large", "fusion_vit_base"]:
        model = Image_Tabular_Concat_ViT_Model(
            model_name=cfg.img_model,
            input_vector_dim=feature_dim,
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
    else:
        raise ValueError("Unrecognized Model Name!")

    if cfg.combine_crossval is True:
        test_labels_list = []

    for ckpt_id, ckpt_path in enumerate(cfg.ckpts_list):
        if cfg.combine_crossval is True:
            dataloader = all_dataloaders[ckpt_id]

        state_dict = torch.load(ckpt_path, map_location="cpu")

        msg = model.load_state_dict(state_dict, strict=False)
        accelerator.print(msg)

        model = accelerator.prepare(model)

        model.eval()
        assert not model.training

        test_preds_proba, _, test_labels = test_utils.predict_proba_pytorch(
            dataloader,
            model,
            1,
            device,
            accelerator,
            cfg.data_type,
            cfg.tab_to_text,
            cfg.model_name,
            phase="test",
            txt_processor=txt_processors,
            context_length=CONTEXT_LENGTH,
            group_age=cfg.group_age,
        )

        test_preds_proba = torch.from_numpy(test_preds_proba).to(device)
        test_labels = test_labels.to(device)

        test_preds_proba_list.append(test_preds_proba)

        if cfg.combine_crossval is True:
            test_labels_list.append(test_labels)

        if test_labels_check is not None:
            assert np.array_equal(
                test_labels_check, test_labels
            ), "test_labels must be the same for every iteration"

        model = accelerator.unwrap_model(model)

        os.makedirs(os.path.join(cfg.save_root, f"ckpt_{ckpt_id}"), exist_ok=True)

        accelerator.save(
            model.state_dict(),
            os.path.join(cfg.save_root, f"ckpt_{ckpt_id}", "best_state.pkl"),
        )
        accelerator.save(
            test_preds_proba,
            os.path.join(cfg.save_root, f"ckpt_{ckpt_id}", "test_preds_proba.pt"),
        )
        accelerator.save(
            test_labels,
            os.path.join(cfg.save_root, f"ckpt_{ckpt_id}", "test_labels.pt"),
        )

    # Evaluate
    if cfg.combine_crossval is True:
        all_test_labels = test_labels_list
    else:
        all_test_labels = test_labels

    if len(test_labels.shape) == 1:  # binary classes or multiclasses
        if n_classes == 2:
            test_eval = compute_binary_metrics_crossval(
                test_preds_proba_list, all_test_labels, device
            )
        elif n_classes > 2:
            test_eval = compute_multiclass_metrics_crossval(
                test_preds_proba_list, all_test_labels, n_classes, device
            )
    else:  # multilabels
        test_eval = compute_multilabel_metrics_crossval(
            test_preds_proba_list, all_test_labels, n_classes, device
        )

    accelerator.print(
        f"Acc: {test_eval['acc']['mean']:.4f} \u00B1 {test_eval['acc']['std']:.4f};",
        f"AUC: {test_eval['auc']['mean']:.4f} \u00B1 {test_eval['auc']['std']:.4f};",
        f"AP: {test_eval['ap']['mean']:.4f} \u00B1 {test_eval['ap']['std']:.4f};",
    )

    if accelerator.is_local_main_process:
        run.track(
            {
                "Accuracy": test_eval["acc"]["mean"],
                "AUROC": test_eval["auc"]["mean"],
                "AP": test_eval["ap"]["mean"],
            },
            context={**context_set, "moment": "mean"},
        )

        run.track(
            {
                "Accuracy": test_eval["acc"]["std"],
                "AUROC": test_eval["auc"]["std"],
                "AP": test_eval["ap"]["std"],
            },
            context={**context_set, "moment": "std"},
        )

    if accelerator.is_local_main_process and not cfg.combine_crossval:
        test_preds_proba_np_list = [
            test_preds_proba.cpu().numpy() for test_preds_proba in test_preds_proba_list
        ]
        test_labels_np = test_labels.cpu().numpy()

        if len(test_labels.shape) == 1:  # binary classes or multiclasses
            if n_classes == 2:
                _compute_metrics_for_roc_crossval = (
                    compute_binary_metrics_for_roc_crossval
                )
                _plot_roc_curve_crossval = plot_roc_curve_crossval

                _compute_metrics_for_pr_crossval = (
                    compute_binary_metrics_for_pr_crossval
                )
                _plot_pr_curve_crossval = plot_pr_curve_crossval

            elif n_classes > 2:
                _compute_metrics_for_roc_crossval = (
                    compute_multiclass_metrics_for_roc_crossval
                )
                _plot_roc_curve_crossval = plot_multiclass_roc_curve_crossval

                _compute_metrics_for_pr_crossval = (
                    compute_multiclass_metrics_for_pr_crossval
                )
                _plot_pr_curve_crossval = plot_multiclass_pr_curve_crossval

        else:  # multilabels
            _compute_metrics_for_roc_crossval = (
                compute_multilabel_metrics_for_roc_crossval
            )
            _plot_roc_curve_crossval = plot_multiclass_roc_curve_crossval

            _compute_metrics_for_pr_crossval = (
                compute_multilabel_metrics_for_pr_crossval
            )
            _plot_pr_curve_crossval = plot_multiclass_pr_curve_crossval

        # ROC curve track
        roc_crossval_metrics = _compute_metrics_for_roc_crossval(
            test_preds_proba_np_list, test_labels_np, device
        )
        roc_fig, roc_ax = plt.subplots()
        roc_fig = _plot_roc_curve_crossval(
            roc_fig, roc_ax, "b", "dimgray", *roc_crossval_metrics
        )
        # plt.show()
        # plt.close(roc_fig)
        aim_track_fig_img(run, roc_fig, fig_name="roc_curve", _context=context_set)

        # PR curve track
        pr_crossval_metrics = _compute_metrics_for_pr_crossval(
            test_preds_proba_np_list, test_labels_np, device
        )
        pr_fig, pr_ax = plt.subplots()
        pr_fig = _plot_pr_curve_crossval(pr_fig, pr_ax, "b", *pr_crossval_metrics)
        # plt.show()
        # plt.close(pr_fig)
        aim_track_fig_img(run, pr_fig, fig_name="pr_curve", _context=context_set)

    # Free memory
    accelerator.clear()

    return run_hash


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--config-path", type=str)
    args = parser.parse_args()

    with open(args.config_path, "r") as file:
        yaml_cfg = yaml.safe_load(file)

    import psutil, time

    p = psutil.Process(os.getppid())
    dt_string = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(p.create_time()))

    config_dict = yaml_cfg["hyperparams"]
    config_dict["save_root"] = os.path.join(
        config_dict["save_root"], str(os.getppid()) + "_" + dt_string
    )
    if isinstance(config_dict["image_size"], list):
        config_dict["image_size"] = tuple(config_dict["image_size"])

    patch_tabular_datasets = [
        "CBIS-DDSM-tfds-with-tabular-2classes",
        "CBIS-DDSM-tfds-with-tabular-2classes-birad3",
        "EMBED-unique-mapping-tfds-with-tabular-2classes",
    ]

    CONTEXT_LENGTH = None

    transform_dict = build_transform_dict(input_size=config_dict["image_size"])
    txt_processors = None

    if config_dict["combine_crossval"]:
        dataset_fold_names = config_dict["dataset"]
        data_fold_dirs = config_dict["datadir"]

        context_list = [{"subset": "train"}, {"subset": "val"}, {"subset": "test"}]
        run_hash = None

        # Iterate through train/val/test splits
        split_id = -1
        for dataset_split_name, data_split_dir in zip(
            dataset_fold_names, data_fold_dirs
        ):
            split_id += 1

            # Iterate through each fold in the current split
            fold_id = -1

            all_dataset_folds_list = []

            for dataset_name, data_dir in zip(dataset_split_name[0], data_split_dir[0]):
                fold_id += 1

                dataset_name_list = [[]] * 3
                data_dir_list = [[]] * 3

                dataset_name_list[split_id].append(dataset_name)
                data_dir_list[split_id].append(data_dir)

                combined_datasets = get_combined_datasets(
                    dataset_name_list[0],
                    dataset_name_list[1],
                    dataset_name_list[2],
                    transform_dict,
                    data_dir_list[0],
                    data_dir_list[1],
                    data_dir_list[2],
                )

                all_dataset_folds_list.extend(combined_datasets[split_id])

            config_dict["save_path"] = []
            run_hash = test_ann_pytorch(*args)

    else:
        dataset_names = config_dict["dataset"]
        data_dirs = config_dict["datadir"]

        combined_datasets = get_combined_datasets(
            dataset_names[0],
            dataset_names[1],
            dataset_names[2],
            transform_dict,
            data_dirs[0],
            data_dirs[1],
            data_dirs[2],
        )
        all_train_datasets, all_val_datasets, all_test_datasets = combined_datasets
        if len(all_train_datasets) != 0:
            train_dataset = CustomConcatDataset(all_train_datasets)
        val_dataset = CustomConcatDataset(all_val_datasets)
        test_dataset = CustomConcatDataset(all_test_datasets)

        datasets = [train_dataset, val_dataset, test_dataset]

        dataset_idx = None

        ckpts_list = config_dict["ckpts_list"]

        run_desc = config_dict["run_desc"]

        config_dict["save_path"] = []

        context_list = [{"subset": "train"}, {"subset": "val"}, {"subset": "test"}]

        run_hash = None
        for dataset_name, data_dir, dataset, context_name in zip(
            dataset_names, data_dirs, datasets, context_list
        ):
            args = (
                config_dict,
                dataset,
                dataset_name,
                context_name,
                run_hash,
                dataset_idx,
                10,
                dataset.get_feature_dim(),
            )
            run_hash = test_ann_pytorch(*args)
