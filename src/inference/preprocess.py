"""Hand localization: crop an image tightly to the detected hand."""
import mediapipe as mp
import numpy as np
from PIL import Image, ImageOps

_mp_hands = mp.solutions.hands


def crop_hand(img, margin=0.25, min_confidence=0.5):
    """Return (cropped_image, found). Falls back to the full image if no hand is found."""
    rgb = np.array(img.convert("RGB"))
    h, w = rgb.shape[:2]

    with _mp_hands.Hands(static_image_mode=True, max_num_hands=1,
                         min_detection_confidence=min_confidence) as hands:
        result = hands.process(rgb)

    if not result.multi_hand_landmarks:
        return img, False

    lm = result.multi_hand_landmarks[0].landmark
    xs = [p.x * w for p in lm]
    ys = [p.y * h for p in lm]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)

    # Square box around the hand, plus a margin, so the hand is not stretched
    side = max(x1 - x0, y1 - y0) * (1 + margin)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    box = tuple(int(v) for v in (cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2))
    return img.crop(box), True

def limit_size(img, max_side=640):
    """Shrink large photos so hand detection is fast and reliable."""
    img = ImageOps.exif_transpose(img).convert("RGB")
    img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return img