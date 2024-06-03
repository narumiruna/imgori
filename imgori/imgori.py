import torch
from torch import nn
from torchvision.transforms._presets import ImageClassification

from .nn import mobilenet_v3
from .typing import Orientation
from .typing import PILImage
from .utils import download_url

DEFAULT_MODEL = (
    "https://github.com/narumiruna/imgori/releases/download/v0.2.7-mobilenet-v3/imgori_mobilenet_v3_small.pth"  # noqa
)


class Imgori:
    model: nn.Module
    transform: ImageClassification

    def __init__(
        self,
        model_path: str | None = None,
        device: torch.device | str = "cpu",
    ):
        self.device = device
        self.transform = ImageClassification(crop_size=224, resize_size=256)

        self.model = self.load_model(model_path or DEFAULT_MODEL)

    def load_model(self, model_path: str) -> nn.Module:
        if model_path.startswith(("s3://", "http://", "https://")):
            model_path = download_url(model_path)

        state_dict = torch.load(model_path, map_location=self.device)

        model = mobilenet_v3(num_classes=len(Orientation))
        model.load_state_dict(state_dict["model"])
        model.eval()
        return model.to(self.device)

    @torch.no_grad()
    def __call__(self, img: PILImage) -> Orientation:
        img_tensor = self.transform(img)
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        output = self.model(img_tensor)
        output = output.argmax(dim=1).item()
        return Orientation(output)
