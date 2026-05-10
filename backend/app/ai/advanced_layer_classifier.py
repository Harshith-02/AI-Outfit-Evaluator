def classify_outfit_layers(

    segmentation_items,

    clothing_data,

    colors
):

    primary_outerwear = (
        "Unknown"
    )

    inner_layer = (
        "Unknown"
    )

    layer_style = (
        "Minimal"
    )

    upper = clothing_data[
        "upper_wear"
    ]

    lower = clothing_data[
        "lower_wear"
    ]

    # =========================
    # STREETWEAR LAYERING
    # =========================

    if (

        "upper_clothes"
        in segmentation_items

        and

        lower == "denim blue"
    ):

        primary_outerwear = (
            "Denim Jacket"
        )

        inner_layer = (
            "White T-Shirt"
        )

        layer_style = (
            "Streetwear Layered Fit"
        )

    # =========================
    # FORMAL LAYERING
    # =========================

    elif (

        upper == "black"

        and

        lower == "white"
    ):

        primary_outerwear = (
            "Formal Blazer"
        )

        inner_layer = (
            "Dress Shirt"
        )

        layer_style = (
            "Luxury Formal Layering"
        )

    # =========================
    # MONOCHROME
    # =========================

    elif upper in [

        "black",

        "gray",

        "dark gray"
    ]:

        primary_outerwear = (
            "Monochrome Outerwear"
        )

        inner_layer = (
            "Neutral Base Layer"
        )

        layer_style = (
            "Minimal Street Style"
        )

    # =========================
    # FALLBACK
    # =========================

    else:

        primary_outerwear = (
            "Casual Outerwear"
        )

        inner_layer = (
            "Standard Inner Layer"
        )

        layer_style = (
            "Casual Styling"
        )

    return {

        "primary_outerwear":
            primary_outerwear,

        "inner_layer":
            inner_layer,

        "layer_style":
            layer_style
    }