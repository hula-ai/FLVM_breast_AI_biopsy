import torch.nn as nn
import torch
import torch.nn.functional as F
import timm

from torchvision import models


class Image_Tabular_Concat_ViT_Model(nn.Module):
    def __init__(
        self,
        model_name,
        input_vector_dim,
        num_classes,
        drop_rate,
        drop_path_rate,
        img_emb_dim,
        img_proj_dim,
        vec_proj_dim,
        hidden_dim,
        tab_encoder_layers,
        freeze_backbone,
        use_pretrained=True,
    ):

        super(Image_Tabular_Concat_ViT_Model, self).__init__()
        self.model_name = model_name
        self.freeze_backbone = freeze_backbone

        self.vit = timm.create_model(
            model_name,
            pretrained=use_pretrained,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
        )

        self.img_emb_proj = nn.Linear(img_emb_dim, img_proj_dim)

        self.vec_emb_proj = self.build_tabular_encoder(
            input_vector_dim, vec_proj_dim, tab_encoder_layers
        )

        self.fc1 = nn.Linear(img_proj_dim + vec_proj_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

        self.dropout_layer = nn.Dropout(p=drop_rate)

    def build_tabular_encoder(self, input_size, embedding_dim, num_layers):
        modules = [nn.Linear(input_size, embedding_dim)]
        for _ in range(num_layers - 1):
            modules.extend([nn.ReLU(), nn.Linear(embedding_dim, embedding_dim)])
        return nn.Sequential(*modules)

    def forward(self, image, vector_data):
        if self.freeze_backbone:
            with torch.no_grad():
                x1 = self.vit.forward_features(image)
        else:
            x1 = self.vit.forward_features(image)
        x2 = vector_data.float()

        x1 = F.relu(self.img_emb_proj(x1[:, 0]))

        x2 = F.relu(self.vec_emb_proj(x2))

        if len(x1.shape) == 1:
            x1 = torch.unsqueeze(x1, 0)

        x = torch.cat((x1, x2), dim=1)
        x = F.relu(self.fc1(x))

        x = self.dropout_layer(x)
        x = self.fc2(x)

        return x
