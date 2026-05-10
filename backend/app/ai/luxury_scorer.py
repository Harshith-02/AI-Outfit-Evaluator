def calculate_luxury_score(

    clothing_data,

    outfit_score,

    complexity_score
):

    luxury_score = (
        outfit_score
        * 0.6
    ) + (
        complexity_score
        * 0.4
    )

    upper = clothing_data[
        "upper_wear"
    ]

    lower = clothing_data[
        "lower_wear"
    ]

    # =========================
    # PREMIUM COMBINATIONS
    # =========================

    if (
        upper == "black"
        and
        lower == "white"
    ):

        luxury_score += 10

    if lower == "denim blue":

        luxury_score += 5

    luxury_score = min(
        luxury_score,
        100
    )

    if luxury_score >= 90:

        luxury_level = (
            "Luxury Designer"
        )

    elif luxury_score >= 75:

        luxury_level = (
            "Premium Fashion"
        )

    else:

        luxury_level = (
            "Standard Fashion"
        )

    return {

        "luxury_score":
            round(luxury_score, 2),

        "luxury_level":
            luxury_level
    }