from rembg import remove
from PIL import Image
import os

OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def remove_background(image_path):

    input_image = Image.open(image_path)

    output_image = remove(input_image)

    output_path = f"{OUTPUT_DIR}/removed_bg.png"

    output_image.save(output_path)

    return {
        "background_removed": True,
        "output_path": output_path
    }