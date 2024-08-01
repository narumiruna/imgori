import torch
from torch import nn
from torchvision.transforms._presets import ImageClassification

from .models import DEFAULT_MODEL
from .nn import mobilenet_v3
from .types import Orientation
from .types import PILImage
from .utils import download_url
from .utils import timeit


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

    @timeit
    @torch.no_grad()
    def __call__(self, img: PILImage) -> Orientation:
        img_tensor = self.transform(img)
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        output = self.model(img_tensor)
        output = output.argmax(dim=1).item()
        return Orientation(output)
