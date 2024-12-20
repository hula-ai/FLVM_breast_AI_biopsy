import torch
from itertools import zip_longest
import numpy as np

def custom_collate(data):

    images = [d['image'] for d in data]
    labels = [d['label'] for d in data]
    image_paths = [d['image_path'] for d in data]
    mask_paths = [d['mask_path'] for d in data]

    images = torch.stack(images)
    labels = torch.tensor(labels)

    return {
        'image': images,
        'label': labels,
        'image_path': image_paths,
        'mask_path': mask_paths
    }


def collate_breast_cancer_patch_tabular(data):

    images = [d['image'] for d in data]
    labels = [d['label'] for d in data]
    image_paths = [d['image_path'] for d in data]

    feat_vecs = [d['feat_vec'] for d in data]
    feat_vec_names = [d['feat_vec_names'] for d in data]

    feat_vecs = np.array(list(zip_longest(*feat_vecs, fillvalue=None))).T.tolist()
    feat_vec_names = np.array(list(zip_longest(*feat_vec_names, fillvalue=None))).T.tolist()

    patients = [d['patient'] for d in data]
    breast_densities = [d['breast_density'] for d in data]
    breast_sides = [d['breast_side'] for d in data]
    abnormals = [d['abnormal'] for d in data]
    mass_shapes = [d['mass_shape'] for d in data]
    mass_margins = [d['mass_margin'] for d in data]
    mass_denses = [d['mass_dense'] for d in data]
    calc_morphs = [d['calc_morph'] for d in data]
    calc_dists = [d['calc_dist'] for d in data]

    marital_statuses = [d['marital_status'] for d in data]
    ethnicities = [d['ethnicity'] for d in data]
    ethnic_groups = [d['ethnic_group'] for d in data]
    ages = [d['age'] for d in data]
    findings = [d['findings'] for d in data]
    birads = [d['birad'] for d in data]
    dataset_names = [d['dataset_name'] for d in data]


    images = torch.stack(images)
    labels = torch.tensor(labels)
    ages = torch.tensor(ages)

    

    return {
        'image': images,
        'label': labels,
        'image_path': image_paths,
        'feat_vec': feat_vecs,
        'feat_vec_names': feat_vec_names,

        'patient': patients,
        'breast_density': breast_densities,
        'breast_side': breast_sides,
        'abnormal': abnormals,
        'mass_shape': mass_shapes,
        'mass_margin': mass_margins,
        'mass_dense': mass_denses,
        'calc_morph': calc_morphs,
        'calc_dist': calc_dists,

        'marital_status': marital_statuses,
        'ethnicity': ethnicities,
        'ethnic_group': ethnic_groups,
        'age': ages,
        'findings': findings,
        'birad': birads,
        'dataset_name': dataset_names
    }