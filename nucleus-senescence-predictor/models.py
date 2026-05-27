import torch
from torch import nn
import timm
from torchvision import transforms

import pandas as pd
import csv
import numpy as np

import utils
import time

from transformers import AutoModel

# !! necessary for some models used during development - discarded later, bcs it didnt work.
from huggingface_hub import login

HF_TOKEN = ""
if HF_TOKEN:
    login(token=HF_TOKEN)
# -


class CustomXception(nn.Module):
    """Xception with Spatial Dropout2d on feature maps + standard Dropout before FC."""

    def __init__(self, output_bins=1, dropout_prob=0.5):
        super(CustomXception, self).__init__()
        self.base_model = timm.create_model(
            "legacy_xception", pretrained=True, num_classes=0
        )
        in_features = self.base_model.num_features

        self.spatial_dropout = nn.Dropout2d(p=dropout_prob)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Dropout(dropout_prob), nn.Linear(in_features, output_bins)
        )

    def forward(self, x):
        x = self.base_model.forward_features(x)
        x = self.spatial_dropout(x)  # Drop entire feature maps
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class CustomConvNeXt(nn.Module):
    def __init__(self, output_bins=1, dropout_prob=0.5):
        super(CustomConvNeXt, self).__init__()
        # self.base_model = timm.create_model('convnext_base', pretrained=True, num_classes=0)
        self.base_model = timm.create_model(
            "convnext_small", pretrained=True, num_classes=0
        )

        in_features = self.base_model.num_features

        # in_features = self.base_model.get_classifier().in_features
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Add a custom classifier with Dropout and Linear layers
        self.fc = nn.Sequential(
            nn.Dropout(dropout_prob), nn.Linear(in_features, output_bins)
        )

    def forward(self, x):
        x = self.base_model.forward_features(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def get_model(model, out_bins, dropout_rate, FREEZE_BACKBONE=False, custom=True):

    if model == "xception" and custom == False:
        model = timm.create_model(
            "legacy_xception", pretrained=True, num_classes=out_bins
        )

    elif model == "xception":
        model = CustomXception(output_bins=out_bins, dropout_prob=dropout_rate)

    elif model == "convnext":
        model = CustomConvNeXt(output_bins=out_bins, dropout_prob=dropout_rate)

    else:
        model = timm.create_model(model, pretrained=True, drop_rate=dropout_rate)
        if hasattr(model, "head"):
            # check if head has a nested fc layer (e.g. ConvNeXt)
            if hasattr(model.head, "fc"):
                model.head.fc = nn.Linear(model.head.fc.in_features, out_bins)
            else:
                model.head = nn.Linear(model.head.in_features, out_bins)
            # -
            # model.head = nn.Linear(model.head.in_features, out_bins)
        elif hasattr(model, "classifier"):
            model.classifier = nn.Linear(model.classifier.in_features, out_bins)
        elif hasattr(model, "fc"):
            model.fc = nn.Linear(model.fc.in_features, out_bins)

    return model
