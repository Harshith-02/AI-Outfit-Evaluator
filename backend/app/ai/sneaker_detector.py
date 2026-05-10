def detect_sneaker_style(

    clothing_data,

    style
):

    footwear = clothing_data[
        "footwear"
    ]

    sneaker_type = (
        "Unknown"
    )

    sneaker_score = 50

    # =========================
    # WHITE SNEAKERS
    # =========================

    if footwear == "white":

        sneaker_type = (
            "Classic White Sneakers"
        )

        sneaker_score += 35

    # =========================
    # STREETWEAR BOOST
    # =========================

    if style == "Streetwear":

        sneaker_score += 15

    sneaker_score = min(
        sneaker_score,
        100
    )

    return {

        "sneaker_type":
            sneaker_type,

        "sneaker_score":
            sneaker_score
    }