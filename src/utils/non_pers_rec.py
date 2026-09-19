"""Module to collect functions and objects for non-personalised recommendations lectures.

The dataset is loaded through `utils.data_utils`, which reads a local copy of the
ml-100k archive when there is one and falls back to the GroupLens URLs otherwise.
"""

import logging

from typing import List

import pandas as pd  # noqa: F401  (kept: the notebooks import pandas through this module)

from .data_utils import load_items, load_train_test, load_users
from .style.colours import Colour

logger = logging.getLogger(__name__)

df_rating, df_rating_test = load_train_test()
df_users = load_users()
df_items = load_items()

# Columns 0 to 4 hold the title, the two dates, the IMDb url and the "unknown"
# flag; the 18 multi-hot genre columns start at "Action". Slicing at 7, as this
# module used to do, silently dropped "Action" and "Adventure".
genre_cols = df_items.columns[5:]

genre_series = df_items[genre_cols].apply(
    lambda row: row[row == 1].index.tolist(), axis=1
)


def recommend_one():
    """Function to return the recommendations based on the most popular criterion."""
    return df_items.loc[[df_items.score.argmax() + 1]]


def recommend_n(n_recommendations: int = 10) -> pd.DataFrame:
    """Function to return recommendations based on the most popular criterion.
    Parameters
    ----------
    n_recommendations: int
        Number of recommendations to return.
        default: 10

    Returns
    -------
    pd.DataFrame
        The dataframe containing the recommendations in order of preference.
    """
    return df_items.sort_values(by="score", ascending=False).iloc[:n_recommendations]


def get_recent_liked_movie(user_id: int, threshold: float = 4.0) -> int:
    """Function to get the id of the movie that the user has liked most recently.

    Parameters
    ----------
    user_id : int
        The id of the user we are interested in.
    threshold : float
        The rating threshold, to determine whether a user appreciated the movie.
        default: 4.0

    Returns
    -------
    int
        The movie id that the user has liked most recently.
    """
    # check if user has liked movie above threshold
    usr_liked_movies = df_rating[
        (df_rating.UserId == user_id) & (df_rating.Rating >= threshold)
    ]

    # if there is no movie the user has liked then recommend the most popular movies.
    if usr_liked_movies.empty:
        logger.warning(
            Colour.RED
            + f"The user has not liked any movie above {threshold}"
            + Colour.ENDC
        )
        return recommend_n()

    # Calculate the max timestamp, to get the last appreciated movies.
    max_timestmp = usr_liked_movies.Timestamp.max()
    # If there is more than one movie at the same timestamp then get the most liked one.
    usr_liked_movies_last = usr_liked_movies[usr_liked_movies.Timestamp == max_timestmp]
    return (
        usr_liked_movies_last.sort_values(by="Rating", ascending=False).iloc[0].MovieId
    )


def get_genre(movie_id: int) -> List[str]:
    """Function to get the genre of the inserted movie.

    Parameters
    ----------
    movie_id : int
        The id of the movie whose genre is to be returned.

    Returns
    -------
    list of str
        the genres of the inserted movie.
    """
    return genre_series.loc[movie_id]
