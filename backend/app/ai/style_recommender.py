def generate_style_recommendations(

    clothing_data,

    style
):

    recommendations = []

    footwear = clothing_data.get(
        "footwear"
    )

    upper = clothing_data.get(
        "upper_wear"
    )

    lower = clothing_data.get(
        "lower_wear"
    )

    # =========================
    # FORMAL STYLE
    # =========================

    if style == "Formal Fashion":

        recommendations.append(

            "A premium wristwatch would "
            "enhance the executive aesthetic"
        )

        recommendations.append(

            "Minimal silver accessories "
            "would improve elegance"
        )

    # =========================
    # STREETWEAR
    # =========================

    elif style == "Streetwear":

        recommendations.append(

            "Layered streetwear styling "
            "works very well here"
        )

        recommendations.append(

            "High-top sneakers could "
            "enhance the urban aesthetic"
        )

        recommendations.append(

            "A monochrome watch or chain "
            "would elevate the outfit"
        )

    # =========================
    # BLACK + WHITE
    # =========================

    if (
        upper == "black"
        and
        lower == "white"
    ):

        recommendations.append(

            "Silver accessories would "
            "complement this monochrome style"
        )

    # =========================
    # WHITE SHOES
    # =========================

    if footwear == "white":

        recommendations.append(

            "White sneakers maintain a "
            "clean modern appearance"
        )

    # =========================
    # DENIM
    # =========================

    if lower == "denim blue":

        recommendations.append(

            "Slim-fit denim enhances the "
            "streetwear silhouette"
        )

    # =========================
    # FALLBACK
    # =========================

    if len(recommendations) == 0:

        recommendations.append(

            "Outfit styling appears "
            "balanced and modern"
        )

    return {

        "style_recommendations":
            recommendations
    }