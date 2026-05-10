from transformers import pipeline

from PIL import Image

import numpy as np

from sklearn.cluster import KMeans


# =========================
# LOAD SEGMENTATION MODEL
# =========================

segmenter = pipeline(

    task="image-segmentation",

    model=
    "mattmdjaga/segformer_b2_clothes",

    device=-1
)


# =========================
# RGB → ADVANCED COLOR AI
# =========================

def rgb_to_color_name(rgb):

    r, g, b = rgb

    brightness = (
        r + g + b
    ) / 3

    # =========================
    # WHITE FAMILY
    # =========================

    if (
        r > 235 and
        g > 235 and
        b > 235
    ):
        return "pure white"

    if (
        r > 220 and
        g > 215 and
        b > 205
    ):
        return "white"

    if (
        r > 200 and
        g > 195 and
        b > 180
    ):
        return "cream"

    # =========================
    # BLACK / GRAY FAMILY
    # =========================

    if brightness < 40:
        return "jet black"

    if brightness < 65:
        return "black"

    if (
        abs(r - g) < 12 and
        abs(g - b) < 12
    ):

        if brightness < 90:
            return "charcoal"

        if brightness < 140:
            return "dark gray"

        if brightness < 185:
            return "gray"

        if brightness > 185:
            return "off white"

        return "silver"

    # =========================
    # BLUE FAMILY
    # =========================

    if (
        b > r and
        abs(r - g) < 40
    ):

        if brightness < 90:
            return "navy"

        if brightness < 130:
            return "denim blue"

        return "sky blue"

    # =========================
    # RED FAMILY
    # =========================

    if (
        r > g + 35 and
        r > b + 35
    ):

        if brightness < 90:
            return "maroon"

        if r > 190:
            return "red"

        return "brick red"

    # =========================
    # GREEN FAMILY
    # =========================

    if (
        g > r + 30 and
        g > b + 30
    ):

        if brightness < 90:
            return "forest green"

        if g > 170:
            return "lime green"

        return "olive green"

    # =========================
    # YELLOW / GOLD
    # =========================

    if (
        r > 180 and
        g > 180 and
        b < 140
    ):

        if brightness > 200:
            return "gold"

        return "mustard"

    # =========================
    # ORANGE
    # =========================

    if (
        r > 190 and
        g > 120 and
        b < 100
    ):
        return "orange"

    # =========================
    # PINK
    # =========================

    if (
        r > 190 and
        b > 150 and
        g < 180
    ):
        return "pink"

    # =========================
    # PURPLE
    # =========================

    if (
        r > 100 and
        b > 100 and
        g < 100
    ):
        return "purple"

    # =========================
    # BROWN FAMILY
    # =========================

    if (
        r > 100 and
        g > 60 and
        b < 70
    ):

        if brightness < 90:
            return "dark brown"

        return "brown"

    # =========================
    # BEIGE / TAN
    # =========================

    if (
        r > 170 and
        g > 150 and
        b > 110
    ):

        if brightness > 210:
            return "sand"

        return "beige"

    # =========================
    # FALLBACK
    # =========================

    return "neutral gray"


# =========================
# ADVANCED COLOR EXTRACTION
# =========================

def extract_color(masked_pixels):

    if len(masked_pixels) == 0:
        return "unknown"

    # Remove dark noise

    masked_pixels = masked_pixels[

        np.mean(
            masked_pixels,
            axis=1
        ) > 35
    ]

    # Remove bright glare

    masked_pixels = masked_pixels[

        np.mean(
            masked_pixels,
            axis=1
        ) < 240
    ]

    if len(masked_pixels) < 50:
        return "unknown"

    # Sample optimization

    if len(masked_pixels) > 4000:

        indices = np.random.choice(

            len(masked_pixels),

            4000,

            replace=False
        )

        masked_pixels = (
            masked_pixels[
                indices
            ]
        )

    # KMeans clustering

    kmeans = KMeans(

        n_clusters=3,

        random_state=42,

        n_init=10
    )

    labels = kmeans.fit_predict(
        masked_pixels
    )

    unique, counts = np.unique(

        labels,

        return_counts=True
    )

    dominant_cluster = unique[
        np.argmax(counts)
    ]

    dominant_rgb = (

        kmeans.cluster_centers_[

            dominant_cluster

        ].astype(int)
    )

    return rgb_to_color_name(
        dominant_rgb
    )


# =========================
# CENTER WEIGHTED EXTRACTION
# =========================

def extract_center_weighted_pixels(

    image_np,

    binary_mask
):

    coords = np.column_stack(
        np.where(binary_mask)
    )

    if len(coords) == 0:
        return np.array([])

    center_y = np.mean(
        coords[:, 0]
    )

    center_x = np.mean(
        coords[:, 1]
    )

    distances = np.sqrt(

        (coords[:, 0] - center_y) ** 2 +

        (coords[:, 1] - center_x) ** 2
    )

    threshold = np.percentile(
        distances,
        70
    )

    filtered_coords = coords[
        distances < threshold
    ]

    if len(filtered_coords) == 0:
        return np.array([])

    masked_pixels = image_np[

        filtered_coords[:, 0],

        filtered_coords[:, 1]
    ]

    return masked_pixels


# =========================
# MAIN AI ANALYZER
# =========================

def analyze_clothing_ai(

    image_path,

    dominant_colors
):

    image = Image.open(
        image_path
    ).convert("RGB")

    image_np = np.array(image)

    results = segmenter(image)

    clothing_items = []

    upper_color = "unknown"

    lower_color = "unknown"

    footwear_color = "unknown"

    outerwear = "Unknown"

    innerwear = "Unknown"

    for item in results:

        label = (
            item["label"]
            .lower()
        )

        clothing_items.append(
            label
        )

        mask = np.array(

            item["mask"]
            .convert("L")
        )

        binary_mask = (
            mask > 127
        )

        mask_area = np.sum(
            binary_mask
        )

        # Ignore tiny masks

        if mask_area < 1500:
            continue

        masked_pixels = (

            extract_center_weighted_pixels(

                image_np,

                binary_mask
            )
        )

        if len(masked_pixels) < 100:
            continue

        color = extract_color(
            masked_pixels
        )

        # =========================
        # UPPER WEAR
        # =========================

        if label in [

            "upper-clothes",

            "upper_clothes",

            "shirt",

            "coat",

            "jacket"
        ]:

            upper_color = color

            outerwear = label

        # =========================
        # LOWER WEAR
        # =========================

        elif label in [

            "pants",

            "skirt",

            "trousers"
        ]:

            lower_color = color

        # =========================
        # FOOTWEAR
        # =========================

        elif label in [

            "left-shoe",

            "right-shoe",

            "left_shoe",

            "right_shoe",

            "shoe",

            "boots",

            "sandals"
        ]:

            if color != "unknown":

                footwear_color = color

    # =========================
    # GLOBAL PALETTE CORRECTION
    # =========================

    global_dark_colors = [

        "black",

        "charcoal",

        "dark gray",

        "gray"
    ]

    # -------------------------
    # Upper wear correction
    # -------------------------

    if (

        upper_color in [

            "sky blue",

            "denim blue"
        ]

        and

        any(

            c in global_dark_colors

            for c in dominant_colors
        )
    ):

        if "black" in dominant_colors:

            upper_color = "black"

        elif "dark gray" in dominant_colors:

            upper_color = "dark gray"

    # -------------------------
    # Lower wear correction
    # -------------------------

    if (

        lower_color in [

            "sky blue",

            "denim blue"
        ]
    ):

        if "white" in dominant_colors:

            lower_color = "white"

        elif "off white" in dominant_colors:

            lower_color = "off white"

        elif "silver" in dominant_colors:

            lower_color = "off white"

        elif "gray" in dominant_colors:

            lower_color = "gray"

        elif "black" in dominant_colors:

            lower_color = "charcoal"

    # -------------------------
    # Footwear correction
    # -------------------------

    if footwear_color == "unknown":

        if "black" in dominant_colors:

            footwear_color = "black"

        elif "white" in dominant_colors:

            footwear_color = "white"

        elif "off white" in dominant_colors:

            footwear_color = "off white"

    # =========================
    # INNER LAYER AI
    # =========================

    if (

        outerwear == "coat"

        and

        upper_color in [

            "black",

            "charcoal",

            "dark gray"
        ]
    ):

        innerwear = (
            "White T-Shirt"
        )

    elif upper_color in [

        "white",

        "pure white",

        "ivory",

        "cream",

        "off white"
    ]:

        innerwear = (
            "Minimal Tee"
        )

    elif upper_color in [

        "denim blue",

        "navy"
    ]:

        innerwear = (
            "Streetwear Base Layer"
        )

    else:

        innerwear = (
            "Standard Innerwear"
        )

    # =========================
    # CLEAN LABEL
    # =========================

    outerwear = (

        outerwear

        .replace("-", " ")

        .replace("_", " ")

        .title()
    )

    # =========================
    # FINAL RESPONSE
    # =========================

    return {

        "upper_wear":
            upper_color,

        "lower_wear":
            lower_color,

        "footwear":
            footwear_color,

        "primary_outerwear":
            outerwear,

        "inner_layer":
            innerwear,

        "detected_items":
            sorted(
                list(
                    set(clothing_items)
                )
            )
    }