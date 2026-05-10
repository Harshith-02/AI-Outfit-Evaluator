def generate_fashion_narration(
    clothing_data,
    outfit_analysis,
    style_analysis
):

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

    style = style_analysis.get(
        "detected_style",
        "Unknown Style"
    )

    score = outfit_analysis.get(
        "outfit_score",
        0
    )

    narration = (

        f"This outfit presents a "
        f"{style.lower()} aesthetic. "

        f"The {upper} upper wear pairs "
        f"with {lower} lower wear "

        f"to create a visually balanced "
        f"appearance. "
    )

    if score >= 90:

        narration += (

            "The outfit achieves a premium "
            "fashion presence with strong "
            "color harmony and elegant contrast. "
        )

    elif score >= 75:

        narration += (

            "The styling feels modern and "
            "well coordinated with balanced "
            "fashion elements. "
        )

    else:

        narration += (

            "The outfit has a casual appearance "
            "but could benefit from stronger "
            "styling coordination. "
        )

    if footwear == "beige":

        narration += (

            "Darker footwear may improve "
            "the formal visual aesthetic."
        )

    return {

        "fashion_narration": narration
    }