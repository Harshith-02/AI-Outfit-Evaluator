def detect_fashion_aesthetic(
    colors,
    style,
    clothing_data=None
):

    upper = "unknown"

    lower = "unknown"

    if clothing_data:

        upper = clothing_data.get(
            "upper_wear",
            "unknown"
        )

        lower = clothing_data.get(
            "lower_wear",
            "unknown"
        )

    # =========================
    # LUXURY MONOCHROME
    # =========================

    if (

        upper == "black"

        and

        lower == "white"
    ):

        aesthetic = (
            "Luxury Monochrome"
        )

    # =========================
    # EXECUTIVE FORMAL
    # =========================

    elif style == "Formal Fashion":

        aesthetic = (
            "Executive Formal"
        )

    # =========================
    # STREETWEAR
    # =========================

    elif style == "Streetwear":

        aesthetic = (
            "Urban Street"
        )

    # =========================
    # MINIMALIST
    # =========================

    elif style == "Minimal Fashion":

        aesthetic = (
            "Minimalist"
        )

    # =========================
    # DEFAULT
    # =========================

    else:

        aesthetic = (
            "Modern Casual"
        )

    return {

        "fashion_aesthetic":
            aesthetic
    }