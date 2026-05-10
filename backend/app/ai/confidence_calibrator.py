def calibrate_style_confidence(

    style_result,

    clothing_data,

    segmentation_items
):

    confidence = style_result[
        "confidence"
    ]

    style = style_result[
        "detected_style"
    ]

    upper = clothing_data[
        "upper_wear"
    ]

    lower = clothing_data[
        "lower_wear"
    ]

    footwear = clothing_data[
        "footwear"
    ]

    # =========================
    # STREETWEAR BOOST
    # =========================

    if style == "Streetwear":

        confidence += 20

        # Denim + sneakers
        if (
            lower == "denim blue"
            and
            footwear == "white"
        ):

            confidence += 15

        # Layered outfit
        if (
            "upper_clothes"
            in segmentation_items
        ):

            confidence += 10

    # =========================
    # FORMAL BOOST
    # =========================

    if style == "Formal Fashion":

        confidence += 15

        if (
            upper == "black"
            and
            lower == "white"
        ):

            confidence += 10

    # =========================
    # LIMIT
    # =========================

    confidence = min(
        confidence,
        99
    )

    return {

        "detected_style":
            style,

        "confidence":
            round(confidence, 2)
    }