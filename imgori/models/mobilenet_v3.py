from typing import Optional

import mlconfig
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights
from torchvision.models import mobilenet_v3_small


@mlconfig.register
def mobilenet_v3(num_classes: Optional[int] = None) -> nn.Module:
    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)

    if num_classes is not None:
        model.classifier[-1] = nn.Linear(model.classifier[3].in_features, num_classes)

    return model
