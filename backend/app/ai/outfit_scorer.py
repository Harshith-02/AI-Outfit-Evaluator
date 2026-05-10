def calculate_outfit_score(
    colors,
    clothing_data=None
):

    # =========================
    # INITIAL SCORE
    # =========================

    score = 50

    recommendations = []

    # =========================
    # COLOR GROUPS
    # =========================

    neutral_colors = [
        "black",
        "white",
        "gray",
        "silver",
        "dark gray",
        "light gray",
        "beige"
    ]

    vibrant_colors = [
        "red",
        "yellow",
        "orange",
        "pink",
        "purple"
    ]

    formal_colors = [
        "black",
        "white",
        "navy",
        "gray",
        "dark gray"
    ]

    # =========================
    # COLOR ANALYSIS
    # =========================

    neutral_count = sum(
        color in neutral_colors
        for color in colors
    )

    vibrant_count = sum(
        color in vibrant_colors
        for color in colors
    )

    formal_count = sum(
        color in formal_colors
        for color in colors
    )

    unique_colors = len(set(colors))

    # =========================
    # SCORING RULES
    # =========================

    # Neutral elegance
    if neutral_count >= 2:
        score += 20

    # Formal styling
    if formal_count >= 3:
        score += 15

    # Black + white combo
    if (
        "black" in colors and
        "white" in colors
    ):
        score += 15

        recommendations.append(
            "Classic black and white combination works very well"
        )

    # Navy + white
    if (
        "navy" in colors and
        "white" in colors
    ):
        score += 10

    # Beige + brown
    if (
        "brown" in colors and
        "beige" in colors
    ):
        score += 10

    # Too many vibrant colors
    if vibrant_count > 2:

        score -= 15

        recommendations.append(
            "Too many bright colors reduce outfit balance"
        )

    # Too many different colors
    if unique_colors > 5:

        score -= 10

        recommendations.append(
            "Simpler color combinations may improve the outfit"
        )

    # =========================
    # CLOTHING-BASED LOGIC
    # =========================

    if clothing_data:

        upper = clothing_data.get(
            "upper_wear",
            "unknown"
        )

        lower = clothing_data.get(
            "lower_wear",
            "unknown"
        )

        footwear = clothing_data.get(
            "footwear",
            "unknown"
        )

        # Monochrome styling
        if upper == lower:

            score += 5

            recommendations.append(
                "Monochrome styling creates a clean aesthetic"
            )

        # Formal contrast
        if (
            upper == "black" and
            lower == "white"
        ):

            score += 10

            recommendations.append(
                "Excellent contrast between upper and lower wear"
            )

        # Footwear matching
        if footwear == "beige":

            recommendations.append(
                "Darker footwear may improve visual contrast"
            )

        if footwear == "white":

            recommendations.append(
                "White footwear gives a modern casual look"
            )

        if footwear == "black":

            score += 5

    # =========================
    # FINAL SCORE LIMIT
    # =========================

    score = max(
        0,
        min(score, 100)
    )

    # =========================
    # STYLE CLASSIFICATION
    # =========================

    if score >= 90:

        style = "Luxury Fashion"

    elif score >= 80:

        style = "Premium Fashion"

    elif score >= 70:

        style = "Stylish Casual"

    elif score >= 55:

        style = "Balanced Outfit"

    else:

        style = "Needs Improvement"

    # =========================
    # OCCASION DETECTION
    # =========================

    if formal_count >= 3:

        occasion = "Formal / Business"

    elif vibrant_count >= 2:

        occasion = "Party Wear"

    else:

        occasion = "Casual Wear"

    # =========================
    # CONFIDENCE SCORE
    # =========================

    confidence_score = min(
        95,
        70 + (neutral_count * 5)
    )

    # =========================
    # FALLBACK RECOMMENDATION
    # =========================

    if len(recommendations) == 0:

        recommendations.append(
            "Outfit looks visually balanced"
        )

    # =========================
    # FINAL RESPONSE
    # =========================

    return {

        "outfit_score": score,

        "confidence_score": confidence_score,

        "style": style,

        "occasion": occasion,

        "recommendations": recommendations
    }