import torch
from torch import nn
from torch.hub import load_state_dict_from_url
from torchvision.transforms._presets import ImageClassification

from .nn import mobilenet_v3
from .typing import Orientation
from .typing import PathLike
from .typing import PILImage
from .utils import load_state_dict_from_s3_url

DEFAULT_MODEL = (
    "https://github.com/narumiruna/imgori/releases/download/v0.2.1-mobilenet-v3/imgori_mobilenet_v3_small.pth"  # noqa
)


def load_model(model_path: PathLike, device: torch.device | str = "cpu") -> nn.Module:
    model_path = str(model_path)

    if model_path.startswith("https://"):
        state_dict = load_state_dict_from_url(model_path, map_location=device, progress=True)
    elif model_path.startswith("s3://"):
        state_dict = load_state_dict_from_s3_url(model_path, map_location=device)
    else:
        state_dict = torch.load(model_path, map_location=device)

    model = mobilenet_v3(num_classes=len(Orientation))
    model.load_state_dict(state_dict["model"])
    model.eval()
    return model.to(device)


class Imgori:
    def __init__(
        self,
        model_path: PathLike = DEFAULT_MODEL,
        device: torch.device | str = "cpu",
    ):
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
