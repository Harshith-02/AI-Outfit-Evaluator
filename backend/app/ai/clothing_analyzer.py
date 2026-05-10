from PIL import Image

import numpy as np

from sklearn.cluster import KMeans


# =========================
# RGB → COLOR NAME
# =========================

def rgb_to_color_name(rgb):

    r, g, b = rgb

    brightness = (
        r + g + b
    ) / 3

    # =========================
    # WHITE
    # =========================

    if (
        r > 210 and
        g > 210 and
        b > 210
    ):
        return "white"

    # =========================
    # CREAM / OFF WHITE
    # =========================

    if (
        r > 185 and
        g > 175 and
        b > 150
    ):
        return "cream"

    # =========================
    # BLACK
    # =========================

    if brightness < 55:
        return "black"

    # =========================
    # DARK GRAY
    # =========================

    if (
        abs(r - g) < 18 and
        abs(g - b) < 18 and
        brightness < 120
    ):
        return "dark gray"

    # =========================
    # LIGHT GRAY
    # =========================

    if (
        abs(r - g) < 20 and
        abs(g - b) < 20
    ):
        return "gray"

    # =========================
    # DENIM BLUE
    # =========================

    if (

        b > r and

        abs(r - g) < 35 and

        b > 70 and

        brightness < 170
    ):

        return "denim blue"

    # =========================
    # NAVY
    # =========================

    if (
        b > r and
        b > g and
        brightness < 90
    ):
        return "navy"

    # =========================
    # BLUE
    # =========================

    if (
        b > r + 20 and
        b > g + 20
    ):
        return "blue"

    # =========================
    # RED
    # =========================

    if (
        r > g + 25 and
        r > b + 25
    ):
        return "red"

    # =========================
    # GREEN
    # =========================

    if (
        g > r + 20 and
        g > b + 20
    ):
        return "green"

    # =========================
    # BROWN
    # =========================

    if (
        r > 100 and
        g > 70 and
        b < 80
    ):
        return "brown"

    # =========================
    # BEIGE
    # =========================

    if (
        r > 170 and
        g > 160 and
        b > 120
    ):
        return "beige"

    # =========================
    # YELLOW
    # =========================

    if (
        r > 180 and
        g > 180 and
        b < 120
    ):
        return "yellow"

    return "gray"


# =========================
# MAIN COLOR EXTRACTION
# =========================

def get_main_color(region):

    pixels = np.array(region).reshape(-1, 3)

    # Remove dark noise
    pixels = pixels[
        np.mean(pixels, axis=1) > 35
    ]

    # Remove very bright noise
    pixels = pixels[
        np.mean(pixels, axis=1) < 245
    ]

    if len(pixels) == 0:
        return "unknown"

    # =========================
    # KMEANS
    # =========================

    kmeans = KMeans(

        n_clusters=5,

        random_state=42,

        n_init=10
    )

    labels = kmeans.fit_predict(
        pixels
    )

    unique, counts = np.unique(

        labels,

        return_counts=True
    )

    dominant_cluster = unique[
        np.argmax(counts)
    ]

    dominant_color = (
        kmeans.cluster_centers_[
            dominant_cluster
        ].astype(int)
    )

    return rgb_to_color_name(
        dominant_color
    )


# =========================
# CLOTHING ANALYSIS
# =========================

def analyze_clothing(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")

    width, height = image.size

    # =========================
    # SMARTER REGIONS
    # =========================

    # Upper torso center
    upper_region = image.crop(

        (
            int(width * 0.28),

            int(height * 0.12),

            int(width * 0.72),

            int(height * 0.45)
        )
    )

    # Jeans / pants region
    lower_region = image.crop(

        (
            int(width * 0.32),

            int(height * 0.50),

            int(width * 0.68),

            int(height * 0.82)
        )
    )

    # Shoes only
    footwear_region = image.crop(

        (
            int(width * 0.34),

            int(height * 0.88),

            int(width * 0.66),

            int(height * 0.98)
        )
    )

    # =========================
    # DETECT COLORS
    # =========================

    upper_color = get_main_color(
        upper_region
    )

    lower_color = get_main_color(
        lower_region
    )

    footwear_color = get_main_color(
        footwear_region
    )

    return {

        "upper_wear":
            upper_color,

        "lower_wear":
            lower_color,

        "footwear":
            footwear_color
    }