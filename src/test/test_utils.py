import torch
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm
from MultiMEDal_multimodal_medical.src.datasets.preprocessing.prompt_factory import (
    tab2prompt_breast_lesion,
)


@torch.no_grad()
def predict_pytorch(
    X,
    model,
    device,
    accelerator,
    data_type,
    tab_to_text,
    model_name,
    phase,
    txt_processors,
    context_length,
    group_age,
):
    all_outputs = torch.tensor([], device=device)
    all_labels = torch.tensor([], device=device, dtype=torch.int32)

    for it, data_info in enumerate(X):
        if data_type == "image":
            inputs = data_info["image"]
        elif data_type == "tabular":
            inputs = data_info["feat_vec"]
        elif data_type == "image_tabular":
            image_inputs = data_info["image"]
            feat_inputs = data_info["feat_vec"]

            text_inputs = []
            if tab_to_text:
                text_samples, _ = tab2prompt_breast_lesion(
                    model_name,
                    phase,
                    data_info,
                    txt_processors,
                    _context_length=context_length,
                    _group_age=group_age,
                )

        labels = data_info["label"]

        if data_type in ["image", "tabular"]:
            predictions = model(inputs)
        elif data_type in ["image_tabular"]:
            if tab_to_text:
                if model_name == "pubmed_clip":
                    predictions = model(
                        image_inputs, text_samples[:, :77]
                    )  # maximum seq len for positional embeddings
                else:
                    predictions = model(image_inputs, text_samples)
            else:
                predictions = model(image_inputs, feat_inputs)

        predictions, labels = accelerator.gather((predictions, labels))
        outputs = F.softmax(predictions, dim=1)

        all_outputs = torch.cat((all_outputs, outputs), dim=0)
        all_labels = torch.cat((all_labels, labels), dim=0)

    return all_outputs.detach().cpu(), all_labels


@torch.no_grad()
def predict_proba_pytorch(
    X,
    model,
    num_samples,
    device,
    accelerator,
    data_type,
    tab_to_text,
    model_name,
    phase,
    txt_processor,
    context_length,
    group_age,
):
    all_preds = []
    labels = None
    for _ in range(num_samples):
        preds, labels = predict_pytorch(
            X,
            model,
            device,
            accelerator,
            data_type,
            tab_to_text,
            model_name,
            phase,
            txt_processor,
            context_length,
            group_age,
        )
        all_preds.append(preds)

    return np.stack(all_preds).mean(axis=0), np.stack(all_preds).std(axis=0), labels


@torch.no_grad()
def predict_class_pytorch(
    X,
    model,
    num_samples,
    device,
    accelerator,
    data_type,
    tab_to_text,
    model_name,
    phase,
    txt_processor,
    context_length,
    group_age,
):
    proba_preds, _, labels = predict_proba_pytorch(
        X,
        model,
        num_samples,
        device,
        accelerator,
        data_type,
        tab_to_text,
        model_name,
        phase,
        txt_processor,
        context_length,
        group_age,
    )

    return np.argmax(proba_preds, axis=1), labels
