def analyze_outfit_complexity(

    segmentation_items,

    clothing_data
):

    complexity_score = 55

    complexity_level = (
        "Basic Styling"
    )

    # =========================
    # MULTIPLE ITEMS
    # =========================

    if len(segmentation_items) >= 6:

        complexity_score += 15

    # =========================
    # FOOTWEAR
    # =========================

    if (
        "left_shoe"
        in segmentation_items
    ):

        complexity_score += 10

    # =========================
    # ACCESSORIES
    # =========================

    if (
        "hat"
        in segmentation_items
        or
        "scarf"
        in segmentation_items
    ):

        complexity_score += 10

    # =========================
    # STREETWEAR
    # =========================

    if (
        clothing_data[
            "lower_wear"
        ] == "denim blue"
    ):

        complexity_score += 10

    # =========================
    # LEVEL
    # =========================

    if complexity_score >= 90:

        complexity_level = (
            "Elite Styling"
        )

    elif complexity_score >= 75:

        complexity_level = (
            "Advanced Styling"
        )

    elif complexity_score >= 65:

        complexity_level = (
            "Layered Fashion"
        )

    return {

        "complexity_score":
            complexity_score,

        "complexity_level":
            complexity_level
    }