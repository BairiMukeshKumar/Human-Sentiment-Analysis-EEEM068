# Import PyTorch neural network module.
# This is used to build custom model classes.
import torch.nn as nn

# Import torchvision pretrained models such as ResNet18 and ResNet50.
import torchvision.models as tv_models

# Import timm library for additional modern image models such as Vision Transformer.
import timm


class ResNetClassifier(nn.Module):
    # Custom ResNet-based classifier for 3-class sentiment classification.
    # It can be used for both face crops and full-image classification.

    def __init__(
        self,
        model_name="resnet18",
        num_classes=3,
        pretrained=True,
        freeze_backbone=False
    ):
        # Initialise the parent nn.Module class.
        super().__init__()

        # Load a ResNet18 backbone if selected.
        if model_name == "resnet18":
            # Use ImageNet pretrained weights if pretrained=True.
            weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
            self.backbone = tv_models.resnet18(weights=weights)

        # Load a ResNet50 backbone if selected.
        elif model_name == "resnet50":
            # Use ImageNet pretrained weights if pretrained=True.
            weights = tv_models.ResNet50_Weights.DEFAULT if pretrained else None
            self.backbone = tv_models.resnet50(weights=weights)

        # Stop the script if an unsupported ResNet model is requested.
        else:
            raise ValueError("Supported: resnet18, resnet50")

        # Get the number of features produced by the original ResNet final layer.
        in_features = self.backbone.fc.in_features

        # Remove the original ResNet classification layer.
        # The backbone will now output feature vectors instead of ImageNet classes.
        self.backbone.fc = nn.Identity()

        # If freeze_backbone=True, stop ResNet backbone weights from updating.
        # This is useful for transfer learning where only the classifier head is trained.
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Add a new custom classifier head for the project's sentiment classes.
        # The final output size is num_classes = 3: negative, neutral, positive.
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # Pass input image through ResNet backbone to extract features.
        features = self.backbone(x)

        # Pass extracted features through the custom classifier head.
        return self.classifier(features)


class TimmImageClassifier(nn.Module):
    # Generic image classifier using models from the timm library.
    # This can be used for architectures such as Vision Transformer.

    def __init__(
        self,
        model_name="vit_base_patch16_224",
        num_classes=3,
        pretrained=True,
        freeze_backbone=True
    ):
        # Initialise the parent nn.Module class.
        super().__init__()

        # Create a pretrained timm model without its original classification layer.
        # num_classes=0 makes the model output feature vectors.
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0
        )

        # Get the number of output features produced by the backbone.
        in_features = self.backbone.num_features

        # Freeze the backbone if required.
        # This means only the new classifier head will be trained.
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Add a custom classifier head for 3-class sentiment classification.
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.40),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # Extract high-level image features using the timm backbone.
        features = self.backbone(x)

        # Classify the extracted features into sentiment classes.
        return self.classifier(features)


class FusionMLP(nn.Module):
    # Small neural network for combining outputs from multiple models.
    # It can combine features such as:
    # - face model probabilities,
    # - full-image model probabilities,
    # - number of detected faces,
    # - face detection confidence.

    def __init__(self, input_dim=8, num_classes=3):
        # Initialise the parent nn.Module class.
        super().__init__()

        # Define a small multi-layer perceptron for fusion classification.
        # input_dim=8 is suitable for combining:
        # 3 face probabilities + 3 full-image probabilities + number of faces + confidence.
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes)
        )

    def forward(self, x):
        # Pass the combined feature vector through the fusion network.
        return self.net(x)