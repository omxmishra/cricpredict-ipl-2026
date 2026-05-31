from cricpredict.features import (
    build_feature_vector,
    get_team_features,
    get_h2h_features,
    get_venue_features
)


def predict_match(
    model,
    model_df,
    team1,
    team2,
    venue,
    toss_winner
):

    X = build_feature_vector(
        team1,
        team2,
        venue,
        toss_winner,
        model_df
    )

    probabilities = model.predict_proba(X)[0]

    team1_stats = get_team_features(
        team1,
        model_df
    )

    team2_stats = get_team_features(
        team2,
        model_df
    )

    h2h_stats = get_h2h_features(
        team1,
        team2,
        model_df
    )

    venue_stats = get_venue_features(
        venue,
        model_df
    )

    return {

        "team1": team1,
        "team2": team2,

        "team1_probability":
            round(probabilities[1] * 100, 2),

        "team2_probability":
            round(probabilities[0] * 100, 2),

        "predicted_winner":
            team1
            if probabilities[1] >
               probabilities[0]
            else team2,

        "team1_stats":
            team1_stats,

        "team2_stats":
            team2_stats,

        "h2h_stats":
            h2h_stats,

        "venue_stats":
            venue_stats,

        "feature_vector":
            X
    }