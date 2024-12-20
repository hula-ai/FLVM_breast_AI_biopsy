import numpy as np

from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from sklearn.metrics import precision_recall_curve, average_precision_score


# Set ggplot styles and update Matplotlib with them.
ggplot_styles = {
    "axes.edgecolor": "white",
    "axes.facecolor": "EBEBEB",
    "axes.grid": True,
    "axes.grid.which": "both",
    "axes.spines.left": False,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.spines.bottom": False,
    "grid.color": "white",
    "grid.linewidth": "1.2",
    "xtick.color": "555555",
    "xtick.major.bottom": True,
    "xtick.minor.bottom": False,
    "ytick.color": "555555",
    "ytick.major.left": True,
    "ytick.minor.left": False,
}

plt.rcParams.update(ggplot_styles)


def plot_pr_curve(precision, recall, ap):
    fig, ax = plt.subplots()

    plt.plot(recall, precision, label=r"PR curve (AP = %0.2f)" % (ap))

    plt.ylabel("Precision")
    plt.xlabel("Recall")

    # Set minor ticks/gridline cadence.
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    # Turn minor gridlines on and make them thinner.
    ax.grid(which="minor", linewidth=0.8)

    plt.legend(loc="best")
    plt.tight_layout()

    plt.show()


def plot_roc_curve(fpr, tpr, auroc):
    fig, ax = plt.subplots()

    plt.plot(fpr, tpr, label=r"ROC curve (AUC = %0.2f)" % (auroc))

    plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")

    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")

    # Set minor ticks/gridline cadence.
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    # Turn minor gridlines on and make them thinner.
    ax.grid(which="minor", linewidth=0.8)

    plt.legend(loc="best")
    plt.tight_layout()

    plt.show()


def plot_pr_curve_crossval(
    fig, ax, avg_pr_color,
    precisions, recalls, aps, y_real, y_proba,
    class_name=None, plot_all_curves=True
):
    # fig, ax = plt.subplots()

    num_folds = len(recalls)
    for fold_id, (rec, prec, ap) in enumerate(zip(recalls, precisions, aps)):
        if num_folds > 1:
            label_text = r"PR curve %d (AP = %0.2f)" % (fold_id, ap)
        else:
            if class_name is not None:
                label_text = f"{class_name} (AP = %0.2f)" % (ap)
            else:
                label_text = "PR curve (AP = %0.2f)" % (ap)

        if plot_all_curves:     
            plt.plot(rec, prec, label=label_text, lw=1)
        
    if num_folds > 1:
        precision, recall, _ = precision_recall_curve(y_real, y_proba)

        if class_name is not None:
            _label = f"{class_name} (AP = %0.2f)" % (average_precision_score(y_real, y_proba))
        else:
            _label = r"Precision-Recall (AP = %0.2f)" % (average_precision_score(y_real, y_proba))

        plt.plot(
            recall,
            precision,            
            color=avg_pr_color,
            label=_label,
            lw=2,
            alpha=0.8,
        )

    axis_fontsize = 16

    plt.ylabel("Precision", fontsize=axis_fontsize)
    plt.xlabel("Recall", fontsize=axis_fontsize)

    # Set minor ticks/gridline cadence.
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    # change the fontsize for xticks and yticks
    ax.tick_params(axis='x', labelsize=axis_fontsize)
    ax.tick_params(axis='y', labelsize=axis_fontsize)

    # Turn minor gridlines on and make them thinner.
    ax.grid(which="minor", linewidth=0.8)

    plt.legend(loc="best", prop={'size': axis_fontsize})
    plt.tight_layout()

    # plt.show()

    return fig


def plot_roc_curve_crossval(
    fig, ax, mean_color, std_color,
    tprs, aucs, mean_fpr, mean_tpr, mean_auc, std_auc, tprs_upper, tprs_lower,
    plot_random_roc=True, class_name=None, plot_all_curves=True
):
    # fig, ax = plt.subplots()

    num_folds = len(tprs)

    for fold_id, (tpr, auroc) in enumerate(zip(tprs, aucs)):
        if num_folds > 1:
            label_text = r"ROC curve %d (AUC = %0.2f)" % (fold_id, auroc)
        else:
            if class_name is not None:
                label_text = f"{class_name} (AUC = %0.2f)" % (auroc)
            else:
                label_text = "ROC curve (AUC = %0.2f)" % (auroc)

        if plot_all_curves:
            plt.plot(
                mean_fpr, tpr, label=label_text, lw=1
            )

    
    if num_folds > 1:
        if class_name is not None:
            _label = f"{class_name} (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc)
        else:
            _label = r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc)
        ax.plot(
            mean_fpr,
            mean_tpr,
            color=mean_color,
            label=_label,
            lw=2,
            alpha=0.8,
        )

        ax.fill_between(
            mean_fpr,
            tprs_lower,
            tprs_upper,
            color=std_color,
            alpha=0.2,
            # label=r"$\pm$ 1 std. dev.",
        )

    if plot_random_roc:
        plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")

    axis_fontsize = 16

    plt.ylabel("True Positive Rate", fontsize=axis_fontsize)
    plt.xlabel("False Positive Rate", fontsize=axis_fontsize)

    # Set minor ticks/gridline cadence.
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    # change the fontsize for xticks and yticks
    ax.tick_params(axis='x', labelsize=axis_fontsize)
    ax.tick_params(axis='y', labelsize=axis_fontsize)

    # Turn minor gridlines on and make them thinner.
    ax.grid(which="minor", linewidth=0.8)

    plt.legend(loc="best", prop={'size': axis_fontsize})
    plt.tight_layout()

    # plt.show()

    return fig


def plot_multiclass_pr_curve_crossval(
    fig, ax, classes_names,
    precisions_list, recalls_list, aps_list, y_real_list, y_proba_list
):
    cmap = [plt.cm.tab20, plt.cm.tab20b]
    n_classes = len(classes_names)
    colors = [cmap[0](np.linspace(0, 1, n_classes-n_classes//2)),
                  cmap[1](np.linspace(0, 1, n_classes//2))]
    
    idx = 0
    for precisions, recalls, aps, y_real, y_proba in zip(precisions_list, recalls_list, aps_list, y_real_list, y_proba_list):
        avg_pr_color = colors[idx%2][idx//2]
        plot_pr_curve_crossval(
            fig, ax, avg_pr_color,
            precisions, recalls, aps, y_real, y_proba,
            class_name=classes_names[idx], plot_all_curves=False
        )
        idx += 1
    return fig


def plot_multiclass_roc_curve_crossval(
    fig, ax, classes_names,
    tprs_list, aucs_list, mean_fpr_list, mean_tpr_list, mean_auc_list, std_auc_list, tprs_upper_list, tprs_lower_list
):
    cmap = [plt.cm.tab20, plt.cm.tab20b]
    n_classes = len(classes_names)
    colors = [cmap[0](np.linspace(0, 1, n_classes-n_classes//2)),
                  cmap[1](np.linspace(0, 1, n_classes//2))]
    
    idx = 0
    for tprs, aucs, mean_fpr, mean_tpr, mean_auc, std_auc, tprs_upper, tprs_lower in zip(tprs_list, aucs_list, mean_fpr_list, mean_tpr_list, mean_auc_list, std_auc_list, tprs_upper_list, tprs_lower_list):
        mean_color = colors[idx%2][idx//2]
        std_color = colors[idx%2][idx//2]
        plot_roc_curve_crossval(
            fig, ax, mean_color, std_color,
            tprs, aucs, mean_fpr, mean_tpr, mean_auc, std_auc, tprs_upper, tprs_lower,
            plot_random_roc=(idx==len(tprs_list)-1),
            class_name=classes_names[idx], plot_all_curves=False
        )
        idx += 1 

    return fig
