"""Single-image and batch prediction for the ResNet50 sign classifier."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision import models
from src.inference.preprocess import crop_hand, limit_size

class SignPredictor:
    def __init__(self, checkpoint_path, device=None, autocontrast=False):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.autocontrast = autocontrast  # off by default to match training data

        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.img_size = ckpt["img_size"]
        self.class_to_letter = {int(k): v for k, v in ckpt["class_to_letter"].items()}

        self.model = self._build(ckpt["model_name"], ckpt["num_classes"])
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval().to(self.device)

        self.mean = torch.tensor(ckpt["imagenet_mean"], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(ckpt["imagenet_std"], device=self.device).view(1, 3, 1, 1)

    def to_training_format(self, img):
            """Convert any image to a 28x28 grayscale tensor in [0, 1].
    
            Matches the Sign MNIST training data: grayscale, square, 28x28.
            ImageOps.pad keeps the aspect ratio so the hand is not stretched.
            """
            img = ImageOps.exif_transpose(img).convert("L")
            img = ImageOps.pad(img, (28, 28), method=Image.Resampling.BILINEAR, color=0)
            if self.autocontrast:
                img = ImageOps.autocontrast(img)
            return torch.tensor(list(img.getdata()), dtype=torch.float32).view(1, 1, 28, 28) / 255.0

    @staticmethod
    def _build(name, num_classes):
        # Same architecture as training; weights come from the checkpoint
        if name == "resnet50":
            m = models.resnet50(weights=None)
            m.fc = nn.Linear(m.fc.in_features, num_classes)
            return m
        raise ValueError(f"Unsupported model: {name}")

    def _prepare(self, x):
        # Same steps as prepare_batch in training: resize, 3 channels, ImageNet normalization
        x = x.to(self.device)
        x = F.interpolate(x, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)
        x = x.repeat(1, 3, 1, 1)
        return (x - self.mean) / self.std

    @torch.no_grad()
    def predict(self, images, top_k=3):
        """Predict a list of PIL images. Returns one dict per image."""
        batch = torch.cat([self.to_training_format(img) for img in images])
        probs = F.softmax(self.model(self._prepare(batch)), dim=1).cpu()

        results = []
        for p in probs:
            top_p, top_i = p.topk(top_k)
            results.append({
                "letter": self.class_to_letter[int(top_i[0])],
                "confidence": float(top_p[0]),
                "top_k": [(self.class_to_letter[int(i)], round(float(v), 4))
                          for v, i in zip(top_p, top_i)],
            })
        return results