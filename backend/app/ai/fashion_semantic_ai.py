def analyze_fashion_semantics(

    clothing,

    style,

    aesthetic,

    complexity
):

    outerwear = (
        "Minimal Outerwear"
    )

    footwear = (
        "Lifestyle Sneakers"
    )

    fit_type = (
        "Regular Fit"
    )

    fashion_identity = (
        "Modern Casual"
    )

    upper = clothing[
        "upper_wear"
    ]

    lower = clothing[
        "lower_wear"
    ]

    detected_style = style[
        "detected_style"
    ]

    aesthetic_type = aesthetic[
        "fashion_aesthetic"
    ]

    complexity_level = complexity[
        "complexity_level"
    ]

    # =========================
    # STREETWEAR
    # =========================

    if (
        detected_style
        == "Streetwear"
    ):

        fashion_identity = (
            "Urban Streetwear"
        )

        fit_type = (
            "Oversized Fit"
        )

        if upper in [

            "dark gray",

            "charcoal",

            "black"
        ]:

            outerwear = (
                "Oversized Streetwear Hoodie"
            )

        else:

            outerwear = (
                "Minimal Streetwear Jacket"
            )

        if lower in [

            "charcoal",

            "dark gray",

            "black"
        ]:

            footwear = (
                "Chunky Street Sneakers"
            )

    # =========================
    # FORMAL
    # =========================

    elif (
        detected_style
        == "Formal Fashion"
    ):

        fashion_identity = (
            "Executive Luxury"
        )

        fit_type = (
            "Tailored Fit"
        )

        outerwear = (
            "Luxury Formal Blazer"
        )

        footwear = (
            "Premium Leather Shoes"
        )

    # =========================
    # LUXURY MINIMAL
    # =========================

    elif (
        aesthetic_type
        == "Minimal Luxury"
    ):

        fashion_identity = (
            "Minimal Luxury"
        )

        fit_type = (
            "Slim Minimal Fit"
        )

        outerwear = (
            "Minimalist Overshirt"
        )

        footwear = (
            "Luxury Minimal Sneakers"
        )

    # =========================
    # COMPLEXITY BOOST
    # =========================

    if (
        complexity_level
        == "Elite Styling"
    ):

        fashion_identity += (
            " Elite"
        )

    return {

        "semantic_outerwear":
            outerwear,

        "semantic_footwear":
            footwear,

        "fit_type":
            fit_type,

        "fashion_identity":
            fashion_identity
    }