def calculate_elo_probability(
    team1_elo,
    team2_elo
):
    """
    Calculate win probability using Elo ratings.
    """

    return 1 / (
        1 +
        10 ** (
            (team2_elo - team1_elo) / 400
        )
    )


def get_latest_elo(
    team_name,
    model_df
):
    """
    Return latest Elo rating for a team.
    """

    team_rows = model_df[
        model_df["team1"] == team_name
    ]

    if len(team_rows) == 0:
        raise ValueError(
            f"No Elo found for {team_name}"
        )

    return team_rows.iloc[-1]["team1_elo"]