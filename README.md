## Frozen Large-scale Pretrained Vision-Language Models are the Effective Foundational Backbone for Multimodal Breast Cancer Prediction
#### Authors: Hung Q. Vo, Lin Wang, Kelvin K. Wong, Chika F. Ezeana, Xiaohui Yu, Wei Yang, Jenny Chang, Hien V. Nguyen, and Stephen T.C. Wong

## Abstract
Breast cancer is a pervasive global health
concern among women. Leveraging multimodal data from
enterprise patient databases—including Picture Archiving
and Communication Systems (PACS) and Electronic Health
Records (EHRs)—holds promise for improving prediction.
This study introduces a multimodal deep-learning model
leveraging mammogram datasets to evaluate breast cancer prediction. Our approach integrates frozen large-scale
pretrained vision-language models, showcasing superior
performance and stability compared to traditional imagetabular models across two public breast cancer datasets.
The model consistently outperforms conventional full finetuning methods by using frozen pretrained vision-language
models alongside a lightweight trainable classifier. The
observed improvements are significant. In the CBIS-DDSM
dataset, the Area Under the Curve (AUC) increases from
0.867 to 0.902 during validation and from 0.803 to 0.830
for the official test set. Within the EMBED dataset, AUC
improves from 0.780 to 0.805 during validation. In scenarios with limited data, using Breast Imaging-Reporting
and Data System category three (BI-RADS 3) cases, AUC
improves from 0.91 to 0.96 on the official CBIS-DDSM test
set and from 0.79 to 0.83 on a challenging validation set.
This study underscores the benefits of vision-language
models in jointly training diverse image-clinical datasets
from multiple healthcare institutions, effectively addressing
challenges related to non-aligned tabular features. Combining training data enhances breast cancer prediction on
the EMBED dataset, outperforming all other experiments. In
summary, our research emphasizes the efficacy of frozen
large-scale pretrained vision-language models in multimodal breast cancer prediction, offering superior performance and stability over conventional methods, reinforcing their potential for breast cancer prediction.

## Installation
### Required Packages
```
python==3.8.18
pytorch==1.13.1
timm==0.9.1
transformers==4.44.0
huggingface-hub==0.24.5
open-clip-torch==2.23.0
tokenizers==0.19.1
```
### Install using Conda
```sh
conda env create -f virtual_env.yaml
```

## Usage
### For Training
```sh
cd src/train

python -m torch.distributed.launch --nproc_per_node=4 --master_port=6006 \
    train.py --config-path <PATH TO YAML TRAINING CONFIG>
```
### For Testing
```sh
python -m torch.distributed.launch --nproc_per_node=4 --master_port=6006 \
    test.py --config-path <PATH TO YAML TESTING CONFIG>
```
## Models

<iframe src="imgs/multimodal_framework.pdf" width="100%" height="500" frameborder="0" />

## Results

## License

## Acknowledgement

## References
