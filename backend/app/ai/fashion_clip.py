import clip
import torch
from PIL import Image


device = "cuda" if torch.cuda.is_available() else "cpu"

model, preprocess = clip.load(
    "ViT-B/32",
    device=device
)

fashion_labels = [

    "Business Casual",

    "Formal Fashion",

    "Streetwear",

    "Minimal Fashion",

    "Luxury Fashion",

    "Sporty Outfit",

    "Old Money Style",

    "Smart Casual",

    "Party Wear",

    "Vintage Fashion",

    "Monochrome Style"
]


def analyze_fashion_style(image_path):

    image = preprocess(
        Image.open(image_path)
    ).unsqueeze(0).to(device)

    text = clip.tokenize(
        fashion_labels
    ).to(device)

    with torch.no_grad():

        image_features = model.encode_image(image)

        text_features = model.encode_text(text)

        logits_per_image, _ = model(
            image,
            text
        )

        probs = logits_per_image.softmax(
            dim=-1
        ).cpu().numpy()[0]

    best_index = probs.argmax()

    detected_style = fashion_labels[
        best_index
    ]

    confidence = round(
        float(probs[best_index] * 100),
        2
    )

    return {

        "detected_style": detected_style,

        "confidence": confidence
    }