import torchmetrics
import torch
import numpy as np

from sklearn.metrics import auc
from collections import defaultdict
from typing import List


def compute_binary_metrics(preds_proba_tensor, labels_tensor, device):
    """
    Parameters:
    preds_proba_tensor - A tensor of shape (n, 2) contains probabilities
    labels_tensor - A tensor of shape (n,) contains labels
    device - torch device
    """

    acc_metric = torchmetrics.Accuracy(task="binary").to(device)
    # acc = acc_metric(torch.softmax(
    #     preds_tensor, axis=1)[:, 1], labels_tensor)
    # acc = acc_metric(preds_proba_tensor[:, 1], labels_tensor)
    acc = acc_metric(torch.argmax(preds_proba_tensor, axis=1), labels_tensor)

    auroc_metric = torchmetrics.AUROC(task="binary").to(device)
    # auroc = auroc_metric(torch.softmax(
    #     preds_tensor, axis=1)[:, 1], labels_tensor)
    auroc = auroc_metric(preds_proba_tensor[:, 1], labels_tensor)

    roc_metric = torchmetrics.ROC(task="binary").to(device)
    # fpr, tpr, roc_thresholds = roc_metric(torch.softmax(
    #     preds_tensor, axis=1)[:, 1], labels_tensor)
    fpr, tpr, roc_thresholds = roc_metric(preds_proba_tensor[:, 1], labels_tensor)

    ap_metric = torchmetrics.AveragePrecision(task="binary").to(device)
    # ap = ap_metric(torch.softmax(preds_tensor, axis=1)[:, 1], labels_tensor)
    ap = ap_metric(preds_proba_tensor[:, 1], labels_tensor)

    pr_metric = torchmetrics.PrecisionRecallCurve(task="binary").to(device)
    # precision, recall, pr_thresholds = pr_metric(
    #     torch.softmax(preds_tensor, axis=1)[:, 1], labels_tensor)
    precision, recall, pr_thresholds = pr_metric(
        preds_proba_tensor[:, 1], labels_tensor
    )

    return {
        "acc": acc,
        "auroc": auroc,
        "roc": {"fpr": fpr, "tpr": tpr, "thresholds": roc_thresholds},
        "ap": ap,
        "pr": {"precision": precision, "recall": recall, "thresholds": pr_thresholds},
    }


def compute_multiclass_metrics(preds_proba_tensor, labels_tensor, n_classes, device):
    """
    Parameters:
    preds_proba_tensor - A tensor of shape (n, c) contains probabilities
    labels_tensor - A tensor of shape (n,) contains labels
    device - torch device
    """

    acc_metric = torchmetrics.Accuracy(task="multiclass", num_classes=n_classes).to(
        device
    )
    # acc = acc_metric(torch.softmax(
    #     preds_tensor, axis=1)[:, 1], labels_tensor)
    acc = acc_metric(preds_proba_tensor, labels_tensor)

    auroc_metric = torchmetrics.AUROC(task="multiclass", num_classes=n_classes).to(
        device
    )
    # auroc = auroc_metric(torch.softmax(
    #     preds_tensor, axis=1)[:, 1], labels_tensor)
    auroc = auroc_metric(preds_proba_tensor, labels_tensor)

    roc_metric = torchmetrics.ROC(task="multiclass", num_classes=n_classes).to(device)
    # fpr, tpr, roc_thresholds = roc_metric(torch.softmax(
    #     preds_tensor, axis=1)[:, 1], labels_tensor)
    fpr, tpr, roc_thresholds = roc_metric(preds_proba_tensor, labels_tensor)

    ap_metric = torchmetrics.AveragePrecision(
        task="multiclass", num_classes=n_classes
    ).to(device)
    # ap = ap_metric(torch.softmax(preds_tensor, axis=1)[:, 1], labels_tensor)
    ap = ap_metric(preds_proba_tensor, labels_tensor)

    pr_metric = torchmetrics.PrecisionRecallCurve(
        task="multiclass", num_classes=n_classes
    ).to(device)
    # precision, recall, pr_thresholds = pr_metric(
    #     torch.softmax(preds_tensor, axis=1)[:, 1], labels_tensor)
    precision, recall, pr_thresholds = pr_metric(preds_proba_tensor, labels_tensor)

    cm_metric = torchmetrics.ConfusionMatrix(
        task="multiclass", num_classes=n_classes
    ).to(device)
    cm = cm_metric(preds_proba_tensor.to(device), labels_tensor.to(device))

    return {
        "acc": acc,
        "auroc": auroc,
        "roc": {"fpr": fpr, "tpr": tpr, "thresholds": roc_thresholds},
        "ap": ap,
        "pr": {"precision": precision, "recall": recall, "thresholds": pr_thresholds},
        "cm": cm,
    }


def compute_multilabel_metrics(preds_proba_tensor, labels_tensor, n_classes, device):
    """
    Parameters:
    preds_proba_tensor - A tensor of shape (n, c) contains probabilities
    labels_tensor - A tensor of shape (n, c) contains labels
    device - torch device
    """
    acc_metric = torchmetrics.Accuracy(task="multilabel", num_labels=n_classes).to(
        device
    )
    # acc = acc_metric(torch.softmax(
    #     preds_tensor, axis=1)[:, 1], labels_tensor)
    acc = acc_metric(preds_proba_tensor, labels_tensor)

    auroc_metric = torchmetrics.AUROC(task="multilabel", num_labels=n_classes).to(
        device
    )
    # auroc = auroc_metric(torch.softmax(
    #     preds_tensor, axis=1)[:, 1], labels_tensor)
    auroc = auroc_metric(preds_proba_tensor, labels_tensor)

    auroc_plot_metric = torchmetrics.AUROC(task="multilabel", average='none', num_labels=n_classes).to(
        device
    )
    auroc_plot = auroc_plot_metric(preds_proba_tensor, labels_tensor)

    roc_metric = torchmetrics.ROC(task="multilabel", num_labels=n_classes).to(device)
    # fpr, tpr, roc_thresholds = roc_metric(torch.softmax(
    #     preds_tensor, axis=1)[:, 1], labels_tensor)
    fpr, tpr, roc_thresholds = roc_metric(preds_proba_tensor, labels_tensor)

    ap_metric = torchmetrics.AveragePrecision(
        task="multilabel", num_labels=n_classes
    ).to(device)
    # ap = ap_metric(torch.softmax(preds_tensor, axis=1)[:, 1], labels_tensor)
    ap = ap_metric(preds_proba_tensor, labels_tensor)

    ap_plot_metric = torchmetrics.AveragePrecision(
        task="multilabel", average='none', num_labels=n_classes
    ).to(device)
    # ap = ap_metric(torch.softmax(preds_tensor, axis=1)[:, 1], labels_tensor)
    ap_plot = ap_plot_metric(preds_proba_tensor, labels_tensor)

    pr_metric = torchmetrics.PrecisionRecallCurve(
        task="multilabel", num_labels=n_classes
    ).to(device)
    # precision, recall, pr_thresholds = pr_metric(
    #     torch.softmax(preds_tensor, axis=1)[:, 1], labels_tensor)
    precision, recall, pr_thresholds = pr_metric(preds_proba_tensor, labels_tensor)

    cm_metric = torchmetrics.ConfusionMatrix(
        task="multilabel", num_labels=n_classes
    ).to(device)
    cm = cm_metric(preds_proba_tensor.to(device), labels_tensor.to(device))

    return {
        "acc": acc,
        "auroc": auroc,
        "roc": {"fpr": fpr, "tpr": tpr, "thresholds": roc_thresholds, "aurocs": auroc_plot},
        "ap": ap,
        "pr": {"precision": precision, "recall": recall, "thresholds": pr_thresholds, "aps": ap_plot},
        "cm": cm,
    }


def compute_binary_metrics_crossval(test_preds_proba_folds: list, test_labels, device):
    eval_metrics_folds = defaultdict(list)

    if isinstance(test_labels, list):
        test_preds_proba_folds = [torch.cat(test_preds_proba_folds)]
        test_labels = torch.cat(test_labels)
        

    for fold_id, test_preds_proba in enumerate(test_preds_proba_folds):
        eval_metrics = compute_binary_metrics(
            test_preds_proba, test_labels, device
        )

        eval_metrics_folds["acc"].append(eval_metrics["acc"])
        eval_metrics_folds["auroc"].append(eval_metrics["auroc"])
        eval_metrics_folds["ap"].append(eval_metrics["ap"])

    mean_acc, std_acc = torch.mean(torch.stack(eval_metrics_folds["acc"])), torch.std(
        torch.stack(eval_metrics_folds["acc"])
    )
    mean_auc, std_auc = torch.mean(torch.stack(eval_metrics_folds["auroc"])), torch.std(
        torch.stack(eval_metrics_folds["auroc"])
    )
    mean_ap, std_ap = torch.mean(torch.stack(eval_metrics_folds["ap"])), torch.std(
        torch.stack(eval_metrics_folds["ap"])
    )

    # print("Acc: %0.2f \u00B1 %0.2f" % (mean_acc.item(), std_acc.item()))
    # print("AUC: %0.2f \u00B1 %0.2f" % (mean_auc.item(), std_auc.item()))
    # print("AP: %0.2f \u00B1 %0.2f" % (mean_ap.item(), std_ap.item()))

    return {
        "acc": {"mean": mean_acc, "std": std_acc},
        "auc": {"mean": mean_auc, "std": std_auc},
        "ap": {"mean": mean_ap, "std": std_ap},
        "cm": eval_metrics_folds["cm"]
    }


def compute_multiclass_metrics_crossval(
    test_preds_proba_folds: list, test_labels, n_classes, device
):
    eval_metrics_folds = defaultdict(list)

    if isinstance(test_labels, list):
        test_preds_proba_folds = [torch.cat(test_preds_proba_folds)]
        test_labels = torch.cat(test_labels)

    for fold_id, test_preds_proba in enumerate(test_preds_proba_folds):

        eval_metrics = compute_multiclass_metrics(
            test_preds_proba,
            test_labels,
            n_classes,
            device,
        )

        eval_metrics_folds["acc"].append(eval_metrics["acc"])
        eval_metrics_folds["auroc"].append(eval_metrics["auroc"])
        eval_metrics_folds["ap"].append(eval_metrics["ap"])
        eval_metrics_folds["cm"].append(eval_metrics["cm"])

    mean_acc, std_acc = torch.mean(torch.stack(eval_metrics_folds["acc"])), torch.std(
        torch.stack(eval_metrics_folds["acc"])
    )
    mean_auc, std_auc = torch.mean(torch.stack(eval_metrics_folds["auroc"])), torch.std(
        torch.stack(eval_metrics_folds["auroc"])
    )
    mean_ap, std_ap = torch.mean(torch.stack(eval_metrics_folds["ap"])), torch.std(
        torch.stack(eval_metrics_folds["ap"])
    )

    # print("Acc: %0.2f \u00B1 %0.2f" % (mean_acc.item(), std_acc.item()))
    # print("AUC: %0.2f \u00B1 %0.2f" % (mean_auc.item(), std_auc.item()))
    # print("AP: %0.2f \u00B1 %0.2f" % (mean_ap.item(), std_ap.item()))
    # for cm in eval_metrics_folds["cm"]:
    #     print(cm)

    return {
        "acc": {"mean": mean_acc, "std": std_acc},
        "auc": {"mean": mean_auc, "std": std_auc},
        "ap": {"mean": mean_ap, "std": std_ap},
        "cm": eval_metrics_folds["cm"]
    }


def compute_multilabel_metrics_crossval(
    test_preds_proba_folds: list, test_labels, n_classes, device
):
    eval_metrics_folds = defaultdict(list)

    if isinstance(test_labels, list):
        test_preds_proba_folds = [torch.cat(test_preds_proba_folds)]
        test_labels = torch.cat(test_labels)

    for fold_id, test_preds_proba in enumerate(test_preds_proba_folds):
        eval_metrics = compute_multilabel_metrics(
            test_preds_proba,
            test_labels,
            n_classes,
            device
        )

        eval_metrics_folds["acc"].append(eval_metrics["acc"])
        eval_metrics_folds["auroc"].append(eval_metrics["auroc"])
        eval_metrics_folds["ap"].append(eval_metrics["ap"])
        eval_metrics_folds["cm"].append(eval_metrics["cm"])

    mean_acc, std_acc = torch.mean(torch.stack(eval_metrics_folds["acc"])), torch.std(
        torch.stack(eval_metrics_folds["acc"])
    )
    mean_auc, std_auc = torch.mean(torch.stack(eval_metrics_folds["auroc"])), torch.std(
        torch.stack(eval_metrics_folds["auroc"])
    )
    mean_ap, std_ap = torch.mean(torch.stack(eval_metrics_folds["ap"])), torch.std(
        torch.stack(eval_metrics_folds["ap"])
    )

    # print("Acc: %0.2f \u00B1 %0.2f" % (mean_acc.item(), std_acc.item()))
    # print("AUC: %0.2f \u00B1 %0.2f" % (mean_auc.item(), std_auc.item()))
    # print("AP: %0.2f \u00B1 %0.2f" % (mean_ap.item(), std_ap.item()))
    # for cm in eval_metrics_folds["cm"]:
    #     print(cm)

    return {
        "acc": {"mean": mean_acc, "std": std_acc},
        "auc": {"mean": mean_auc, "std": std_auc},
        "ap": {"mean": mean_ap, "std": std_ap},
        "cm": eval_metrics_folds["cm"]
    }


def compute_binary_metrics_for_roc_crossval(test_preds_proba_folds: list, test_labels, device):
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    for test_preds_proba in test_preds_proba_folds:
        # eval_metrics = compute_binary_metrics(all_preds_tensor, all_labels_tensor, device)
        # test_preds_proba = history['test_preds_proba'][fold_idx]

        eval_metrics = compute_binary_metrics(
            torch.from_numpy(test_preds_proba), torch.from_numpy(test_labels), device
        )

        interp_tpr = np.interp(
            mean_fpr, eval_metrics["roc"]["fpr"], eval_metrics["roc"]["tpr"]
        )
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(eval_metrics["auroc"])

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

    return tprs, aucs, mean_fpr, mean_tpr, mean_auc, std_auc, tprs_upper, tprs_lower


def convert_multiclass_into_ovr_preds(test_preds_proba_folds: list, test_labels):
    # convert multiclass predictions into binary OvR predictions
    test_preds_proba_folds_ovr = []
    test_labels_list_ovr = []

    all_classes = np.unique(test_labels)
    n_classes = all_classes.shape[0]

    for class_id in all_classes.tolist():
        test_preds_proba_fold = []
        test_labels_fold = test_labels.copy()
        
        test_labels_fold[test_labels == class_id] = 1
        test_labels_fold[test_labels != class_id] = 0

        for test_preds_proba in test_preds_proba_folds:
            binary_test_preds_proba = np.zeros((test_preds_proba.shape[0], 2))
            binary_test_preds_proba[:, 1] = test_preds_proba[:, class_id]

            neg_cls_ind_arr = np.ones(n_classes, bool)
            neg_cls_ind_arr[class_id] = False

            binary_test_preds_proba[:, 0] = np.sum(test_preds_proba[:, neg_cls_ind_arr], axis=1)

            test_preds_proba_fold.append(binary_test_preds_proba)


        test_preds_proba_folds_ovr.append(test_preds_proba_fold)
        test_labels_list_ovr.append(test_labels_fold)

    return test_preds_proba_folds_ovr, test_labels_list_ovr


def compute_multiclass_metrics_for_roc_crossval(test_preds_proba_folds: list, test_labels, device):
    test_preds_proba_folds_ovr, test_labels_list_ovr = convert_multiclass_into_ovr_preds(test_preds_proba_folds, test_labels)
    
    # Calculate cross-val roc metrics for every OvR class
    tprs_list, aucs_list = [], []
    mean_fpr_list, mean_tpr_list, mean_auc_list, std_auc_list = [], [], [], []
    tprs_upper_list, tprs_lower_list = [], []

    for _test_preds_proba_folds, _test_labels in zip(test_preds_proba_folds_ovr, test_labels_list_ovr):
        tprs, aucs, mean_fpr, mean_tpr, mean_auc, std_auc, tprs_upper, tprs_lower = \
            compute_binary_metrics_for_roc_crossval(_test_preds_proba_folds, _test_labels, device)

        tprs_list.append(tprs)
        aucs_list.append(aucs)
        mean_fpr_list.append(mean_fpr)
        mean_tpr_list.append(mean_tpr)
        mean_auc_list.append(mean_auc)
        std_auc_list.append(std_auc)
        tprs_upper_list.append(tprs_upper)
        tprs_lower_list.append(tprs_lower)

    return tprs_list, aucs_list, mean_fpr_list, mean_tpr_list, mean_auc_list, std_auc_list, tprs_upper_list, tprs_lower_list


def compute_multilabel_metrics_for_roc_crossval(test_preds_proba_folds: list, test_labels, n_classes, device):
    tprs_list = [[] for _ in range(n_classes)]
    aucs_list = [[] for _ in range(n_classes)]
    mean_fpr_list = [np.linspace(0, 1, 100) for _ in range(n_classes)]

    for test_preds_proba in test_preds_proba_folds:

        eval_metrics = compute_multilabel_metrics(
            test_preds_proba, test_labels, n_classes, device
        )

        for class_id in range(n_classes):

            interp_tpr = np.interp(
                mean_fpr_list[class_id], eval_metrics["roc"]["fpr"][class_id].cpu(), eval_metrics["roc"]["tpr"][class_id].cpu()
            )
            interp_tpr[0] = 0.0
            tprs_list[class_id].append(interp_tpr)
            aucs_list[class_id].append(eval_metrics["roc"]["aurocs"][class_id].cpu())

    mean_tpr_list = [[] for _ in range(n_classes)]
    mean_auc_list = [[] for _ in range(n_classes)]
    std_auc_list = [[] for _ in range(n_classes)]
    tprs_upper_list = [[] for _ in range(n_classes)]
    tprs_lower_list = [[] for _ in range(n_classes)]

    for class_id in range(n_classes):
        mean_tpr_list[class_id] = np.mean(tprs_list[class_id], axis=0)
        mean_tpr_list[class_id][-1] = 1.0
        mean_auc_list[class_id] = auc(mean_fpr_list[class_id], mean_tpr_list[class_id])
        std_auc_list[class_id] = np.std(aucs_list[class_id])

        std_tpr = np.std(tprs_list[class_id], axis=0)
        tprs_upper_list[class_id] = np.minimum(mean_tpr_list[class_id] + std_tpr, 1)
        tprs_lower_list[class_id] = np.maximum(mean_tpr_list[class_id] - std_tpr, 0)


    return tprs_list, aucs_list, mean_fpr_list, mean_tpr_list, mean_auc_list, std_auc_list, tprs_upper_list, tprs_lower_list


def compute_binary_metrics_for_pr_crossval(test_preds_proba_folds: list, test_labels, device):
    y_real = []
    y_proba = []
    precisions = []
    recalls = []
    aps = []

    for test_preds_proba in test_preds_proba_folds:
        # eval_metrics = compute_binary_metrics(all_preds_tensor, all_labels_tensor, device)
        # test_preds_proba = history['test_preds_proba'][fold_idx]

        eval_metrics = compute_binary_metrics(
            torch.from_numpy(test_preds_proba), torch.from_numpy(test_labels), device
        )

        precision = eval_metrics["pr"]["precision"]
        recall = eval_metrics["pr"]["recall"]
        ap = eval_metrics["ap"]

        y_real.append(test_labels)
        y_proba.append(test_preds_proba[:, 1])

        precisions.append(precision)
        recalls.append(recall)
        aps.append(ap)

    y_real = np.concatenate(y_real)
    y_proba = np.concatenate(y_proba)

    return precisions, recalls, aps, y_real, y_proba


def compute_multiclass_metrics_for_pr_crossval(test_preds_proba_folds: list, test_labels, device):
    test_preds_proba_folds_ovr, test_labels_list_ovr = convert_multiclass_into_ovr_preds(test_preds_proba_folds, test_labels)

    # Calculate cross-val roc metrics for every OvR class
    precisions_list, recalls_list, aps_list = [], [], []
    y_real_list, y_proba_list = [], []

    for _test_preds_proba_folds, _test_labels in zip(test_preds_proba_folds_ovr, test_labels_list_ovr):
        precisions, recalls, aps, y_real, y_proba = \
            compute_binary_metrics_for_pr_crossval(_test_preds_proba_folds, _test_labels, device)

        precisions_list.append(precisions)
        recalls_list.append(recalls)
        aps_list.append(aps)
        y_real_list.append(y_real)
        y_proba_list.append(y_proba)

    return precisions_list, recalls_list, aps_list, y_real_list, y_proba_list


def compute_multilabel_metrics_for_pr_crossval(test_preds_proba_folds: list, test_labels, n_classes, device):
    y_real_list = [[] for _ in range(n_classes)]
    y_proba_list = [[] for _ in range(n_classes)]
    precisions_list = [[] for _ in range(n_classes)]
    recalls_list = [[] for _ in range(n_classes)]
    aps_list = [[] for _ in range(n_classes)]

    for test_preds_proba in test_preds_proba_folds:
        # eval_metrics = compute_binary_metrics(all_preds_tensor, all_labels_tensor, device)
        # test_preds_proba = history['test_preds_proba'][fold_idx]

        eval_metrics = compute_multilabel_metrics(
            test_preds_proba, test_labels, n_classes, device
        )

        for class_id in range(n_classes):
            precision = eval_metrics["pr"]["precision"][class_id]
            recall = eval_metrics["pr"]["recall"][class_id]
            ap = eval_metrics["pr"]["aps"][class_id]

            test_labels_class = torch.zeros(test_labels.shape[0])
            test_labels_class[test_labels[:, class_id] == 1] = 1
            y_real_list[class_id].append(test_labels_class.cpu())
            y_proba_list[class_id].append(test_preds_proba[:, class_id].cpu())

            precisions_list[class_id].append(precision)
            recalls_list[class_id].append(recall)
            aps_list[class_id].append(ap)


    for class_id in range(n_classes):            
        y_real_list[class_id] = np.concatenate(y_real_list[class_id])
        y_proba_list[class_id] = np.concatenate(y_proba_list[class_id])

    return precisions_list, recalls_list, aps_list, y_real_list, y_proba_list