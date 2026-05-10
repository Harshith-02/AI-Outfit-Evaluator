from PIL import Image
import numpy as np
from sklearn.cluster import KMeans


def rgb_to_color_name(rgb):

    r, g, b = rgb

    # White / Black / Gray
    if r > 220 and g > 220 and b > 220:
        return "white"

    if r < 40 and g < 40 and b < 40:
        return "black"

    if abs(r - g) < 15 and abs(g - b) < 15:
        if r < 120:
            return "dark gray"
        return "gray"

    # Red shades
    if r > 180 and g < 80 and b < 80:
        return "red"

    if r > 120 and g < 60 and b < 60:
        return "maroon"

    if r > 200 and g > 100 and b > 100:
        return "pink"

    # Orange / Brown
    if r > 200 and 80 < g < 170 and b < 80:
        return "orange"

    if r > 120 and 60 < g < 100 and b < 60:
        return "brown"

    if r > 180 and g > 140 and b > 90:
        return "beige"

    # Yellow / Gold
    if r > 200 and g > 200 and b < 100:
        return "yellow"

    if r > 180 and g > 150 and b < 80:
        return "gold"

    # Green shades
    if g > 150 and r < 120 and b < 120:
        return "green"

    if g > 100 and r < 100 and b > 100:
        return "teal"

    if g > 120 and r > 120 and b < 80:
        return "olive"

    # Blue shades
    if b > 150 and r < 120 and g < 120:
        return "blue"

    if b > 120 and r < 80 and g < 80:
        return "navy"

    if b > 180 and g > 180 and r < 150:
        return "cyan"

    # Purple shades
    if r > 120 and b > 120 and g < 100:
        return "purple"

    if r > 180 and b > 180 and g < 140:
        return "lavender"

    # Neutral tones
    if r > 150 and g > 150 and b > 150:
        return "silver"

    if r > 100 and g > 100 and b > 100:
        return "light gray"

    return "unknown"


def detect_dominant_colors(image_path, num_colors=5):

    image = Image.open(image_path).convert("RGBA")

    image = image.resize((200, 200))

    pixels = np.array(image)

    # Remove transparent pixels
    pixels = pixels[pixels[:, :, 3] > 0]

    # Remove alpha channel
    pixels = pixels[:, :3]

    # Remove very dark noise pixels
    pixels = pixels[np.mean(pixels, axis=1) > 20]

    kmeans = KMeans(
        n_clusters=num_colors,
        random_state=42,
        n_init=10
    )

    kmeans.fit(pixels)

    colors = kmeans.cluster_centers_.astype(int)

    color_names = []

    for color in colors:

        color_name = rgb_to_color_name(color)

        if (
            color_name not in color_names
            and color_name != "unknown"
        ):
            color_names.append(color_name)

    return {
        "dominant_colors": color_names
    }