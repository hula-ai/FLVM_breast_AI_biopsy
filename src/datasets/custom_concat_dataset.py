from torch.utils.data import Dataset, ConcatDataset
from itertools import chain
from typing import Iterable
from torch.utils.data import Subset
from typing import (
    Iterable,
    Sequence
)


class  CustomConcatDataset(ConcatDataset):
    def __init__(self, datasets: Iterable[Dataset]) -> None:
        super().__init__(datasets)

        assert all([datasets[0].classes == d.classes for d in datasets])
        self.classes = datasets[0].classes

    def get_all_images(self):
        return list(chain(*[d.get_all_images() for d in self.datasets]))

    def get_all_lesion_feats(self):
        return list(chain(*[d.get_all_lesion_feats() for d in self.datasets]))        

    def get_all_labels(self):
        return list(chain(*[d.get_all_labels() for d in self.datasets]))        

    def get_lesion_feats_label(self):
        return list(chain(*[d.get_lesion_feats_label() for d in self.datasets]))

    def get_all_patients(self):
        return list(chain(*[d.get_all_patients() for d in self.datasets]))
    
    def get_feature_dim(self):
        datasets_feature_dims = [d.get_feature_dim() for d in self.datasets]
        assert len(set(datasets_feature_dims)) == 1
        return datasets_feature_dims[0]


class CustomSubsetDataset(Subset):
    def __init__(self, dataset: Dataset, indices: Sequence[int]) -> None:
        super().__init__(dataset, indices)

        self.classes = dataset.classes

    def get_all_images(self):
        return [self.dataset.get_all_images()[idx] for idx in self.indices]
    
    def get_all_lesion_feats(self):
        return [self.dataset.get_all_lesion_feats()[idx] for idx in self.indices]
    
    def get_all_labels(self):
        return [self.dataset.get_all_labels()[idx] for idx in self.indices]
    
    def get_lesion_feats_label(self):
        return [self.dataset.get_lesion_feats_label()[idx] for idx in self.indices]
    
    def get_all_patients(self):
        return [self.dataset.get_all_patients()[idx] for idx in self.indices]
    
    def get_feature_dim(self):
        return self.dataset.get_feature_dim()