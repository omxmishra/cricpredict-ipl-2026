import pandas as pd


def get_latest_team_row(
    team_name,
    model_df
):
    """
    Get latest available row for a team.
    """

    team_rows = model_df[
        model_df["team1"] == team_name
    ]

    if len(team_rows) == 0:
        raise ValueError(
            f"No data found for {team_name}"
        )

    return team_rows.iloc[-1]


def get_team_features(
    team_name,
    model_df
):
    """
    Return latest team stats.
    """

    row = get_latest_team_row(
        team_name,
        model_df
    )

    return {
        "elo": row["team1_elo"],
        "form": row["team1_form"],
        "avg_runs": row["team1_avg_runs_last5"],
        "avg_wickets": row["team1_avg_wickets_last5"]
    }


def get_h2h_features(
    team1,
    team2,
    model_df
):
    """
    Get latest H2H statistics.
    """

    matches = model_df[
        (
            (model_df["team1"] == team1)
            &
            (model_df["team2"] == team2)
        )
        |
        (
            (model_df["team1"] == team2)
            &
            (model_df["team2"] == team1)
        )
    ]

    if len(matches) == 0:
        return {
            "team1_h2h": 0.5,
            "team2_h2h": 0.5
        }

    latest = matches.iloc[-1]

    return {
        "team1_h2h":
            latest["team1_h2h_win_pct"],
        "team2_h2h":
            latest["team2_h2h_win_pct"]
    }


def get_venue_features(
    venue,
    model_df
):
    """
    Get venue statistics.
    """

    venue_matches = model_df[
        model_df["venue"] == venue
    ]

    if len(venue_matches) == 0:
        return {
            "venue_avg_score":
                model_df["venue_avg_score"].mean(),
            "venue_match_count": 0
        }

    latest = venue_matches.iloc[-1]

    return {
        "venue_avg_score":
            latest["venue_avg_score"],
        "venue_match_count":
            latest["venue_match_count"]
    }


def build_feature_vector(
    team1,
    team2,
    venue,
    toss_winner,
    model_df
):
    """
    Create feature vector
    expected by trained model.
    """

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

    team1_won_toss = (
        1 if toss_winner == team1
        else 0
    )

    features = pd.DataFrame([
        {
            "team1_elo":
                team1_stats["elo"],

            "team2_elo":
                team2_stats["elo"],

            "team1_form":
                team1_stats["form"],

            "team2_form":
                team2_stats["form"],

            "team1_won_toss":
                team1_won_toss,

            "team1_avg_runs_last5":
                team1_stats["avg_runs"],

            "team2_avg_runs_last5":
                team2_stats["avg_runs"],

            "team1_avg_wickets_last5":
                team1_stats["avg_wickets"],

            "team2_avg_wickets_last5":
                team2_stats["avg_wickets"],

            "team1_h2h_win_pct":
                h2h_stats["team1_h2h"],

            "team2_h2h_win_pct":
                h2h_stats["team2_h2h"],

            "venue_avg_score":
                venue_stats["venue_avg_score"],

            "venue_match_count":
                venue_stats["venue_match_count"]
        }
    ])

    return features