from typing import Optional

import mlconfig
from torch import nn
from torchvision.models import EfficientNet_B0_Weights
from torchvision.models import efficientnet_b0


@mlconfig.register
def efficientnet(num_classes: Optional[int] = None) -> nn.Module:
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

    if num_classes is not None:
        model.classifier[-1] = nn.Linear(model.classifier[1].in_features, num_classes)

    return model
