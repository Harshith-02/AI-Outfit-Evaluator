def analyze_texture_profile(

    clothing_data,

    segmentation_items
):

    texture_score = 60

    texture_type = (
        "Standard Fabric"
    )

    upper = clothing_data[
        "upper_wear"
    ]

    lower = clothing_data[
        "lower_wear"
    ]

    # =========================
    # DENIM
    # =========================

    if lower == "denim blue":

        texture_score += 20

        texture_type = (
            "Premium Denim"
        )

    # =========================
    # MONOCHROME
    # =========================

    if (
        upper == "black"
        and
        lower == "white"
    ):

        texture_score += 15

    # =========================
    # LAYERED
    # =========================

    if (
        "upper_clothes"
        in segmentation_items
    ):

        texture_score += 10

    texture_score = min(
        texture_score,
        100
    )

    return {

        "texture_score":
            texture_score,

        "texture_type":
            texture_type
    }