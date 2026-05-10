def calculate_fashion_metrics(
    colors,
    clothing_data
):

    harmony_score = 82

    contrast_score = 80

    formality_score = 72

    modern_style_score = 78

    upper = clothing_data.get(
        "upper_wear"
    )

    lower = clothing_data.get(
        "lower_wear"
    )

    # Black + White premium combo
    if (
        upper == "black"
        and
        lower == "white"
    ):

        harmony_score = 95

        contrast_score = 96

        formality_score = 94

        modern_style_score = 88

    return {

        "color_harmony":
            harmony_score,

        "contrast_balance":
            contrast_score,

        "formality_score":
            formality_score,

        "modern_style_score":
            modern_style_score
    }