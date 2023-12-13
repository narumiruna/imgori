import torch
from torch import nn
from torchvision.transforms._presets import ImageClassification

from .nn import mobilenet_v3
from .typing import Orientation
from .typing import PathLike
from .typing import PILImage


def load_model(model_path: PathLike, device: torch.device) -> nn.Module:
    model = mobilenet_v3(num_classes=len(Orientation))
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict["model"])
    model.eval()
    return model.to(device)


class Imgori:
    def __init__(self, model_path: PathLike, device: torch.device):
        self.model = load_model(model_path, device)
        self.device = device
        self.transform = ImageClassification(crop_size=224, resize_size=256)

    @torch.no_grad()
    def __call__(self, img: PILImage) -> Orientation:
        img_tensor = self.transform(img)
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        output = self.model(img_tensor)
        output = output.argmax(dim=1).item()
        return Orientation(output)
