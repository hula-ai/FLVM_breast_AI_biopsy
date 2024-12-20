import torch.nn as nn
import torch
import torch.nn.functional as F
import open_clip

from torchvision import models
from transformers import CLIPProcessor, CLIPModel


class Clip_Image_Tabular_Model(nn.Module):
    def __init__(self, model_name, num_classes, drop_rate, freeze_backbone, pretrained_data=None):
        super(Clip_Image_Tabular_Model, self).__init__()
        
        self.model_name = model_name
        if model_name == "flaviagiammarino/pubmed-clip-vit-base-patch32":
            self.model = CLIPModel.from_pretrained(model_name)
        else:
            if pretrained_data is not None:
                self.model, self.preprocess_train, self.preprocess_val = \
                    open_clip.create_model_and_transforms(model_name, pretrained=pretrained_data)
            else:
                self.model, self.preprocess_train, self.preprocess_val = \
                    open_clip.create_model_and_transforms(model_name)
            

        self.freeze_backbone = freeze_backbone

        if model_name in ["EVA02-B-16", "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"]:
            self.proj_layer = nn.Linear(1024, 200)
        elif model_name in ["EVA02-L-14", 'EVA02-L-14-336']:
            self.proj_layer = nn.Linear(1536, 200)
        elif model_name in ['EVA01-g-14', 'EVA01-g-14-plus', 'EVA02-E-14', 'ViT-g-14']:
            self.proj_layer = nn.Linear(2048, 200)
        elif model_name in ['ViT-bigG-14', 'ViT-bigG-14-CLIPA']:
            self.proj_layer = nn.Linear(2560, 200)
        elif model_name in ["flaviagiammarino/pubmed-clip-vit-base-patch32"]:
            self.proj_layer = nn.Linear(1280, 200)

        self.fc1 = nn.Linear(200, 200)
        self.fc2 = nn.Linear(200, num_classes)

        self.dropout_layer = nn.Dropout(p=drop_rate)

    def forward(self, image, text, tabular=None):
        if self.freeze_backbone:
            with torch.no_grad():
                if self.model_name == "flaviagiammarino/pubmed-clip-vit-base-patch32":
                    image_features = self.model.vision_model(image)['pooler_output']
                    text_features = self.model.text_model(text)['pooler_output']                                            
                else:
                    image_features, text_features, logit_scale = self.model(image, text)
        else:
                if self.model_name == "flaviagiammarino/pubmed-clip-vit-base-patch32":
                    image_features = self.model.vision_model(image)['pooler_output']
                    text_features = self.model.text_model(text)['pooler_output']
                else:
                    image_features, text_features, logit_scale = self.model(image, text)

        features = torch.cat((image_features, text_features), dim=1)

        emb_proj = F.relu(self.proj_layer(features))

        x = F.relu(self.fc1(emb_proj))

        x = self.dropout_layer(x)
        x = self.fc2(x)
        
        return x
