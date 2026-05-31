from cricpredict.features import build_feature_vector


def predict_match(
    model,
    model_df,
    team1,
    team2,
    venue,
    toss_winner
):
    """
    Predict match outcome.
    """

    X = build_feature_vector(
        team1=team1,
        team2=team2,
        venue=venue,
        toss_winner=toss_winner,
        model_df=model_df
    )

    probabilities = model.predict_proba(X)[0]

    return {
        "team1": team1,
        "team2": team2,
        "team1_probability": round(
            probabilities[1] * 100,
            2
        ),
        "team2_probability": round(
            probabilities[0] * 100,
            2
        ),
        "predicted_winner": (
            team1
            if probabilities[1] >
               probabilities[0]
            else team2
        )
    }