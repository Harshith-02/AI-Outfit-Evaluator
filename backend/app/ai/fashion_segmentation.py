from transformers import SegformerImageProcessor
from transformers import AutoModelForSemanticSegmentation

from PIL import Image
import torch
import numpy as np


processor = SegformerImageProcessor.from_pretrained(
    "mattmdjaga/segformer_b2_clothes"
)

model = AutoModelForSemanticSegmentation.from_pretrained(
    "mattmdjaga/segformer_b2_clothes"
)


LABELS = {
    0: "background",
    1: "hat",
    2: "hair",
    3: "sunglasses",
    4: "upper_clothes",
    5: "skirt",
    6: "pants",
    7: "dress",
    8: "belt",
    9: "left_shoe",
    10: "right_shoe",
    11: "face",
    12: "left_leg",
    13: "right_leg",
    14: "left_arm",
    15: "right_arm",
    16: "bag",
    17: "scarf"
}


def segment_fashion(image_path):

    image = Image.open(image_path).convert("RGB")

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    outputs = model(**inputs)

    logits = outputs.logits.cpu()

    upsampled_logits = torch.nn.functional.interpolate(
        logits,
        size=image.size[::-1],
        mode="bilinear",
        align_corners=False,
    )

    pred_seg = upsampled_logits.argmax(dim=1)[0]

    unique_classes = np.unique(
        pred_seg.numpy()
    )

    detected_items = []

    for cls in unique_classes:

        label = LABELS.get(int(cls))

        if (
            label
            and label != "background"
        ):
            detected_items.append(label)

    return {
        "detected_clothing_items": detected_items
    }